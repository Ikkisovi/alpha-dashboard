
import { spawn } from 'child_process';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { cacheManager } from '@/lib/cache-manager';
import { ComputeRequest, ComputeResponse } from '@/lib/types';
import Papa from 'papaparse';

export async function POST(request: NextRequest) {
    try {
        const body: ComputeRequest = await request.json();
        const autoIncrement = Boolean(body.auto_increment_factor_ids || body.export_factor_values_csv);
        if (autoIncrement && body.factor_id_offset == null) {
            body.factor_id_offset = computeNextFactorIdOffset();
        }
        const skipCache = Boolean(body.export_factor_values_csv);

        // 1. Generate cache key
        const cacheKey = cacheManager.generateKey(body);

        if (!skipCache) {
            // 2. Check in-memory cache
            const cached = cacheManager.get(cacheKey);
            if (cached) {
                return NextResponse.json({
                    ...cached,
                    cache_hit: true
                });
            }

            // 3. Check filesystem cache
            const fileCached = await cacheManager.getFromFile(cacheKey);
            if (fileCached) {
                cacheManager.set(cacheKey, fileCached); // Promote to memory
                return NextResponse.json({
                    ...fileCached,
                    cache_hit: true
                });
            }
        }

        // 4. Spawn Python computation
        console.log(`Cache miss for ${cacheKey}. Spawning Python...`);
        const result = await runPythonComputation(body);

        if (result.error) {
            return NextResponse.json(result, { status: 500 });
        }

        // 5. Store in cache (memory + file)
        if (!skipCache) {
            cacheManager.set(cacheKey, result);
            await cacheManager.saveToFile(cacheKey, result);
        }

        return NextResponse.json({
            ...result,
            cache_hit: false
        });

    } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Unknown error';
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

async function runPythonComputation(request: ComputeRequest): Promise<ComputeResponse> {
    return new Promise((resolve) => {
        // Resolve python path: Use venv locally, python3 on Vercel
        // process.cwd() in Next.js dev is the dashboard directory
        const venvPython = path.resolve(process.cwd(), '../.venv/bin/python');
        const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3';

        // Path to script (relative to dashboard directory)
        // Try dashboard-local version first (has export_factor_values_csv function)
        let scriptPath = path.resolve(process.cwd(), 'python-backend/compute_service.py');
        if (!fs.existsSync(scriptPath)) {
            // Fallback to parent directory (for backwards compatibility)
            scriptPath = path.resolve(process.cwd(), '../python-backend/compute_service.py');
        }

        const child = spawn(pythonCmd, [scriptPath]);

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        child.on('close', (code) => {
            if (code !== 0) {
                console.error("Python script failed:", stderr);
                resolve({
                    error: "Computation failed",
                    details: stderr,
                    metrics: [],
                    daily_returns: [],
                    computation_time_ms: 0
                });
                return;
            }

            try {
                const json = JSON.parse(stdout);
                resolve(json);
            } catch {
                console.error("Failed to parse Python output:", stdout);
                resolve({
                    error: "Invalid response from backend",
                    details: stdout,
                    metrics: [],
                    daily_returns: [],
                    computation_time_ms: 0
                });
            }
        });

        // Write input
        child.stdin.write(JSON.stringify(request));
        child.stdin.end();

        // Timeout (60 seconds for initial load, subsequent requests are faster due to caching)
        setTimeout(() => {
            child.kill();
            resolve({ error: "Computation timed out (60s limit)", metrics: [], daily_returns: [], computation_time_ms: 0 });
        }, 60000);
    });
}

function readDictCsvMaxId(filePath: string): number | null {
    if (!fs.existsSync(filePath)) return null;
    const text = fs.readFileSync(filePath, 'utf-8');
    const parsed = Papa.parse<Record<string, unknown>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
    }).data;
    let maxId: number | null = null;
    for (const row of parsed) {
        const raw = row.factor_id ?? row.id ?? row.index ?? row.factorId;
        const val = typeof raw === 'number' ? raw : Number(raw);
        if (Number.isFinite(val)) {
            maxId = maxId === null ? val : Math.max(maxId, val);
        }
    }
    return maxId;
}

function readDictJsonMaxId(filePath: string): number | null {
    if (!fs.existsSync(filePath)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        if (!Array.isArray(data)) return null;
        let maxId: number | null = null;
        for (const row of data) {
            const raw = row?.factor_id ?? row?.id ?? row?.index ?? row?.factorId;
            const val = typeof raw === 'number' ? raw : Number(raw);
            if (Number.isFinite(val)) {
                maxId = maxId === null ? val : Math.max(maxId, val);
            }
        }
        return maxId;
    } catch {
        return null;
    }
}

function readFactorValuesHeaderMaxId(filePath: string): number | null {
    if (!fs.existsSync(filePath)) return null;
    const firstLine = fs.readFileSync(filePath, 'utf-8').split(/\r?\n/)[0] || '';
    let maxId: number | null = null;
    const matches = firstLine.matchAll(/factor_(\d+)/g);
    for (const match of matches) {
        const val = Number(match[1]);
        if (Number.isFinite(val)) {
            maxId = maxId === null ? val : Math.max(maxId, val);
        }
    }
    return maxId;
}

function computeNextFactorIdOffset(): number {
    const dataDir = path.join(process.cwd(), 'public', 'data');
    const dictDir = path.join(dataDir, 'factors', 'dictionaries');
    const baseDict = path.join(dataDir, 'factors', 'dictionary.csv');
    const factorValuesDir = path.join(dataDir, 'factor_values');
    let maxId: number | null = null;

    const baseMax = readDictCsvMaxId(baseDict);
    if (baseMax !== null) maxId = maxId === null ? baseMax : Math.max(maxId, baseMax);

    if (fs.existsSync(dictDir)) {
        const files = fs.readdirSync(dictDir).filter((f) => f.endsWith('.csv') || f.endsWith('.json'));
        for (const file of files) {
            const full = path.join(dictDir, file);
            const nextMax = file.endsWith('.json') ? readDictJsonMaxId(full) : readDictCsvMaxId(full);
            if (nextMax !== null) maxId = maxId === null ? nextMax : Math.max(maxId, nextMax);
        }
    }

    if (fs.existsSync(factorValuesDir)) {
        const files = fs.readdirSync(factorValuesDir).filter((f) => f.endsWith('.csv'));
        for (const file of files) {
            const nextMax = readFactorValuesHeaderMaxId(path.join(factorValuesDir, file));
            if (nextMax !== null) maxId = maxId === null ? nextMax : Math.max(maxId, nextMax);
        }
    }

    return (maxId ?? -1) + 1;
}
