import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import Papa from "papaparse";
import { FactorInfo, FactorPnL } from "@/utils/data";

const DATA_DIR = path.join(process.cwd(), "public", "data");
const DICT_DIR = path.join(DATA_DIR, "factors", "dictionaries");
const BASE_DICT_CSV = path.join(DATA_DIR, "factors", "dictionary.csv");
const FACTOR_VALUES_DIR = path.join(DATA_DIR, "factor_values");

function canonicalExpr(expr: string | undefined): string {
  if (!expr) return "";
  const trimmed = expr.replace(/\s+/g, "");
  // Treat scalar multiples as the same factor: Mul(const, inner) -> inner
  const m = trimmed.match(/^Mul\(([-\d\.eE]+),(.*)\)$/);
  if (m) return m[2];
  const m2 = trimmed.match(/^Mul\((.*),([-\d\.eE]+)\)$/);
  if (m2) return m2[1];
  return trimmed;
}

function makeDictKey(d: FactorInfo): string {
  const exprKey = canonicalExpr(d.expr || d.name);
  if (exprKey) return exprKey;
  return `${d.source || "unknown"}:${d.factor_id}`;
}

function parseDictCsv(filePath: string, source: string): FactorInfo[] {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, "utf-8");
  const parsed = Papa.parse<Record<string, any>>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  }).data;
  return parsed
    .filter(r => Object.keys(r).length > 0)
    .map(r => ({
      factor_id: Number(r.factor_id ?? r.id ?? r.index ?? 0),
      name: r.name || `Factor ${r.factor_id ?? ""}`,
      expr: r.expression || r.expr || "",
      description: r.description || "",
      type: r.type || "composite",
      ic: Number(r.ic ?? 0),
      icir: Number(r.icir ?? 0),
      sharpe: Number(r.sharpe ?? 0),
      mdd: Number(r.mdd ?? 0),
      source,
    }));
}

function parseDictJson(filePath: string, source: string): FactorInfo[] {
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    if (!Array.isArray(data)) return [];
    return data.map((r: any, idx: number) => ({
      factor_id: Number(r.factor_id ?? idx),
      name: r.name || `Factor ${r.factor_id ?? idx}`,
      expr: r.expression || r.expr || "",
      description: r.description || "",
      type: r.type || "composite",
      ic: Number(r.ic ?? 0),
      icir: Number(r.icir ?? 0),
      sharpe: Number(r.sharpe ?? 0),
      mdd: Number(r.mdd ?? 0),
      source,
    }));
  } catch {
    return [];
  }
}

function parseReturnsCsv(filePath: string): FactorPnL[] {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, "utf-8");
  const parsed = Papa.parse<Record<string, any>>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  }).data;
  return parsed.filter(r => Object.keys(r).length > 0) as any;
}

export async function GET() {
  try {
    // Load dictionaries from all sources, tracking source file
    type FactorWithSource = FactorInfo & { _sourceFile: string };
    let allFactors: FactorWithSource[] = [];

    if (fs.existsSync(BASE_DICT_CSV)) {
      allFactors.push(...parseDictCsv(BASE_DICT_CSV, "dictionary.csv").map(f => ({ ...f, _sourceFile: "dictionary.csv" })));
    }

    if (fs.existsSync(DICT_DIR)) {
      const files = fs.readdirSync(DICT_DIR).filter(f => f.endsWith(".json") || f.endsWith(".csv"));
      for (const f of files) {
        const full = path.join(DICT_DIR, f);
        if (f.endsWith(".json")) {
          allFactors.push(...parseDictJson(full, f).map(factor => ({ ...factor, _sourceFile: f })));
        } else {
          allFactors.push(...parseDictCsv(full, f).map(factor => ({ ...factor, _sourceFile: f })));
        }
      }
    }

    // Deduplicate by canonical expression while PRESERVING original factor IDs
    // This prevents breaking the factor_id references in pool files
    const dedupedDict: FactorInfo[] = [];
    const seen = new Map<string, FactorInfo>(); // Maps exprKey -> factor (with original ID)
    const factorIdToSource = new Map<number, string>(); // Maps factor_id -> sourceFile for collision detection

    for (const d of allFactors) {
      const exprKey = makeDictKey(d); // Canonical expression key
      const existing = seen.get(exprKey);

      if (!existing) {
        // New unique expression - keep original factor_id

        // Warn if factor_id collision detected from different sources
        const existingSource = factorIdToSource.get(d.factor_id);
        if (existingSource && existingSource !== d._sourceFile) {
          console.warn(`[custom-factors] ID collision: factor_id ${d.factor_id} exists in both ${existingSource} and ${d._sourceFile}`);
        }

        seen.set(exprKey, d);
        dedupedDict.push(d);
        factorIdToSource.set(d.factor_id, d._sourceFile);
      } else {
        // Duplicate expression found - prefer higher IC magnitude when merging
        const existingScore = Math.abs(existing.ic ?? 0);
        const newScore = Math.abs(d.ic ?? 0);
        if (newScore > existingScore) {
          // Update with better metadata, but KEEP original factor_id
          const updatedFactor = { ...d, factor_id: existing.factor_id };
          seen.set(exprKey, updatedFactor);
          const idx = dedupedDict.findIndex(f => f.factor_id === existing.factor_id);
          if (idx >= 0) dedupedDict[idx] = updatedFactor;
        }
      }
    }

    // Load factor values (IDs are already correct - no remapping needed)
    const mergedReturnsMap = new Map<string, FactorPnL & { index?: number }>();

    if (fs.existsSync(FACTOR_VALUES_DIR)) {
      const csvFiles = fs.readdirSync(FACTOR_VALUES_DIR).filter(f => f.endsWith(".csv")).sort();

      for (const csvFile of csvFiles) {
        const csvPath = path.join(FACTOR_VALUES_DIR, csvFile);
        const returns = parseReturnsCsv(csvPath);

        returns.forEach((row) => {
          const date = String((row as any).date || (row as any).Date || "");
          if (!date) return;

          const base = mergedReturnsMap.get(date) || { date };

          // Copy all columns directly (factor IDs are already correctly numbered in CSV)
          Object.keys(row).forEach(k => {
            if (k === "date" || k === "Date") return;
            (base as any)[k] = (row as any)[k];
          });

          mergedReturnsMap.set(date, base);
        });
      }
    }

    const mergedReturns = Array.from(mergedReturnsMap.values())
      .sort((a, b) => String(a.date).localeCompare(String(b.date)))
      .map((r, i) => ({ ...r, index: i }));

    // Log loading stats
    const uniqueSources = new Set(allFactors.map(f => f._sourceFile)).size;
    const idRange = dedupedDict.length > 0
      ? `${Math.min(...dedupedDict.map(f => f.factor_id))}-${Math.max(...dedupedDict.map(f => f.factor_id))}`
      : 'N/A';

    console.log(`[custom-factors] Loaded ${allFactors.length} factors from ${uniqueSources} sources`);
    console.log(`[custom-factors] After deduplication: ${dedupedDict.length} unique factors`);
    console.log(`[custom-factors] Factor ID range: ${idRange} (original IDs preserved)`);
    console.log(`[custom-factors] Returns data: ${mergedReturns.length} dates with ${Object.keys(mergedReturns[0] || {}).filter(k => k.startsWith('factor_')).length} factors`);

    return NextResponse.json({
      dictionary: dedupedDict,
      returns: mergedReturns,
      sources: {
        dictionary: allFactors.map(d => d._sourceFile).filter(Boolean),
        totalFactorsBeforeDedup: allFactors.length,
        uniqueFactorsAfterDedup: dedupedDict.length,
      }
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
