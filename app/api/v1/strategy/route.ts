import { spawn } from 'child_process';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import Papa from 'papaparse';

interface StrategyRequest {
    pool_id?: string;
    factor_ids?: number[] | null;
    train_start?: string;
    train_end?: string;
    test_start?: string;
    test_end?: string;
    target_horizons?: number[];
    strategy?: {
        type?: string;
        top_pct?: number;
        rebalance_days?: number;
    };
    factor_signals?: number[][];  // Optional pre-computed signals
    factor_signal_path?: string;
}

interface PythonRunResult {
    data?: unknown;
    error?: string;
    details?: string;
}

const PYTHON_CMD = resolvePythonCommand();
const STRATEGY_SCRIPT = resolveScriptPath('python-backend/strategy_service.py');
const COMPUTE_SCRIPT = resolveScriptPath('python-backend/compute_service.py');
const SIGNAL_CACHE_DIR = resolveSignalDir();
const SIGNAL_EXTENSION = '.npz';
const STRATEGY_TIMEOUT_MS = 60000;
const COMPUTE_TIMEOUT_MS = 180000;
const DATA_DIR = path.join(process.cwd(), 'public', 'data');
const DICT_DIR = path.join(DATA_DIR, 'factors', 'dictionaries');
const BASE_DICT_CSV = path.join(DATA_DIR, 'factors', 'dictionary.csv');

function resolvePythonCommand(): string {
    const venvPython = path.resolve(process.cwd(), '../.venv/bin/python');
    if (fs.existsSync(venvPython)) return venvPython;
    return 'python3';
}

function resolveScriptPath(relative: string): string {
    const candidates = [
        path.resolve(process.cwd(), `../${relative}`),
        path.resolve(process.cwd(), relative),
    ];
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    return candidates[candidates.length - 1];
}

function resolveSignalDir(): string {
    const candidates = [
        path.resolve(process.cwd(), 'tmp/strategy_signals'),
        path.resolve(process.cwd(), '../tmp/strategy_signals'),
    ];
    for (const dir of candidates) {
        try {
            fs.mkdirSync(dir, { recursive: true });
            return dir;
        } catch (e) {
            console.warn(`[Strategy API] Failed to init signal dir ${dir}: ${e}`);
        }
    }
    throw new Error('Unable to initialize signal cache directory');
}

function runPythonScript(scriptPath: string, payload: unknown, timeoutMs: number, label: string): Promise<PythonRunResult> {
    return new Promise((resolve) => {
        const child = spawn(PYTHON_CMD, [scriptPath]);

        let stdout = '';
        let stderr = '';

        const timer = setTimeout(() => {
            child.kill();
            resolve({ error: `${label} python script timed out (${timeoutMs}ms)` });
        }, timeoutMs);

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ error: `${label} failed to spawn: ${err.message}` });
        });

        child.on('close', (code) => {
            clearTimeout(timer);
            if (code !== 0) {
                console.error(`${label} python script failed:`, stderr);
                resolve({ error: `${label} computation failed`, details: stderr });
                return;
            }

            try {
                const trimmed = stdout.trim();
                let parsed: unknown;
                try {
                    parsed = JSON.parse(trimmed || '{}') as unknown;
                } catch {
                    const lastLine = trimmed.split('\n').filter(Boolean).pop();
                    if (lastLine) {
                        parsed = JSON.parse(lastLine) as unknown;
                    } else {
                        throw new Error('Empty output');
                    }
                }
                resolve({ data: parsed });
            } catch {
                console.error(`${label} failed to parse output:`, stdout);
                resolve({ error: `${label} invalid response from backend`, details: stdout });
            }
        });

        child.stdin.write(JSON.stringify(payload));
        child.stdin.end();
    });
}

function readPoolExpressions(poolId?: string): string[] | null {
    if (!poolId) return null;
    if (!fs.existsSync(poolId)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(poolId, 'utf-8'));
        if (Array.isArray(data?.exprs)) return data.exprs;
    } catch {
        return null;
    }
    return null;
}

function parseDictCsv(filePath: string): Array<{ factorId: number; expr: string }> {
    if (!fs.existsSync(filePath)) return [];
    const text = fs.readFileSync(filePath, 'utf-8');
    const parsed = Papa.parse<Record<string, unknown>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
    }).data;
    return parsed
        .filter((row) => Object.keys(row).length > 0)
        .map((row) => {
            const factorId = Number(row.factor_id ?? row.id ?? row.index ?? row.factorId);
            const expr = String(row.expression ?? row.expr ?? row.Expression ?? '');
            return { factorId, expr };
        })
        .filter((row) => Number.isFinite(row.factorId) && row.expr.length > 0);
}

function parseDictJson(filePath: string): Array<{ factorId: number; expr: string }> {
    try {
        const raw = fs.readFileSync(filePath, 'utf-8');
        const data = JSON.parse(raw);
        if (!Array.isArray(data)) return [];
        return data
            .map((row: Record<string, unknown>) => {
                const factorId = Number(row.factor_id ?? row.id ?? row.index ?? row.factorId);
                const expr = String(row.expression ?? row.expr ?? '');
                return { factorId, expr };
            })
            .filter((row) => Number.isFinite(row.factorId) && row.expr.length > 0);
    } catch {
        return [];
    }
}

function loadFactorExpressionMap(): Map<number, string> {
    const map = new Map<number, string>();
    if (fs.existsSync(BASE_DICT_CSV)) {
        for (const entry of parseDictCsv(BASE_DICT_CSV)) {
            map.set(entry.factorId, entry.expr);
        }
    }
    if (fs.existsSync(DICT_DIR)) {
        const files = fs.readdirSync(DICT_DIR).filter((f) => f.endsWith('.json') || f.endsWith('.csv')).sort();
        for (const file of files) {
            const full = path.join(DICT_DIR, file);
            const entries = file.endsWith('.json') ? parseDictJson(full) : parseDictCsv(full);
            for (const entry of entries) {
                map.set(entry.factorId, entry.expr);
            }
        }
    }
    return map;
}

function buildCustomPoolFile(request: StrategyRequest): string | null {
    if (!request.factor_ids || request.factor_ids.length === 0) return null;
    const poolExprs = readPoolExpressions(request.pool_id);
    const dictMap = loadFactorExpressionMap();
    const exprs: string[] = [];
    let needsCustom = false;

    for (const id of request.factor_ids) {
        const poolExpr = poolExprs && id >= 0 && id < poolExprs.length ? poolExprs[id] : null;
        if (typeof poolExpr === 'string' && poolExpr.length > 0) {
            exprs.push(poolExpr);
            continue;
        }

        const dictExpr = dictMap.get(id);
        if (!dictExpr) {
            throw new Error(`Missing expression for factor_id ${id}`);
        }
        exprs.push(dictExpr);
        needsCustom = true;
    }

    if (!needsCustom) return null;

    const key = buildSignalKey(request);
    const poolPath = path.join(SIGNAL_CACHE_DIR, `${key}.pool.json`);
    if (!fs.existsSync(poolPath)) {
        fs.writeFileSync(poolPath, JSON.stringify({ exprs }, null, 2));
    }
    return poolPath;
}

function extractComputeError(data: unknown): string | undefined {
    if (!data || typeof data !== 'object') return undefined;
    const payload = data as { error?: unknown; traceback?: unknown };
    if (typeof payload.error !== 'string' || payload.error.trim().length === 0) return undefined;
    if (typeof payload.traceback === 'string' && payload.traceback.trim().length > 0) {
        return `${payload.error}\n${payload.traceback}`;
    }
    return payload.error;
}

function buildSignalKey(request: StrategyRequest): string {
    const factorIds = request.factor_ids && request.factor_ids.length > 0
        ? [...new Set(request.factor_ids)].sort((a, b) => a - b)
        : 'all';
    const payload = {
        pool_id: request.pool_id,
        factor_ids: factorIds,
        train_start: request.train_start,
        train_end: request.train_end,
        test_start: request.test_start,
        test_end: request.test_end
    };
    return crypto.createHash('sha1').update(JSON.stringify(payload)).digest('hex');
}

function hasMetricsAndReturns(data: unknown): data is { metrics: unknown[]; daily_returns: unknown[] } {
    if (!data || typeof data !== 'object') return false;
    const payload = data as { metrics?: unknown; daily_returns?: unknown };
    return Array.isArray(payload.metrics) && Array.isArray(payload.daily_returns);
}

async function ensureSignalFile(request: StrategyRequest): Promise<{ signalPath?: string; computeData?: unknown }> {
    if (!request.pool_id) return {};

    const key = buildSignalKey(request);
    const signalPath = path.join(SIGNAL_CACHE_DIR, `${key}${SIGNAL_EXTENSION}`);
    const legacyPath = path.join(SIGNAL_CACHE_DIR, `${key}.npy`);

    if (!fs.existsSync(signalPath) && fs.existsSync(legacyPath)) {
        try {
            fs.rmSync(legacyPath);
        } catch (err) {
            console.warn(`[Strategy API] Failed to remove legacy cache ${legacyPath}:`, err);
        }
    }

    if (!fs.existsSync(signalPath)) {
        const computePayload = {
            pool_id: request.pool_id,
            factor_ids: request.factor_ids && request.factor_ids.length > 0 ? request.factor_ids : null,
            train_start: request.train_start,
            train_end: request.train_end,
            test_start: request.test_start,
            test_end: request.test_end,
            target_horizons: request.target_horizons,
            skip_analytics: true,
            combined_signal_output: signalPath
        };

        const computeResult = await runPythonScript(COMPUTE_SCRIPT, computePayload, COMPUTE_TIMEOUT_MS, '[Compute]');
        if (computeResult.error) {
            throw new Error(computeResult.details ? `${computeResult.error}: ${computeResult.details}` : computeResult.error);
        }
        const computeError = extractComputeError(computeResult.data);
        if (computeError) {
            throw new Error(computeError);
        }

        if (!fs.existsSync(signalPath)) {
            if (hasMetricsAndReturns(computeResult.data)) {
                console.warn('[Strategy API] Combined signal missing; returning compute_service output directly.');
                return { computeData: computeResult.data };
            }
            throw new Error('Combined signal file missing after compute step');
        }
    }

    return { signalPath };
}

export async function POST(request: NextRequest) {
    try {
        const body: StrategyRequest = await request.json();
        const payload: StrategyRequest = { ...body };
        try {
            const customPoolPath = buildCustomPoolFile(payload);
            if (customPoolPath) {
                payload.pool_id = customPoolPath;
                payload.factor_ids = null;
            }
        } catch (err: unknown) {
            const details = err instanceof Error ? err.message : undefined;
            return NextResponse.json({
                error: 'Failed to resolve custom factor expressions',
                details,
                metrics: [],
                daily_returns: []
            }, { status: 500 });
        }

        let signalPath: string | undefined;

        if (payload.pool_id) {
            try {
                const ensured = await ensureSignalFile(payload);
                if (ensured && 'computeData' in ensured && ensured.computeData) {
                    return NextResponse.json(ensured.computeData);
                }
                signalPath = ensured?.signalPath;
                if (signalPath) {
                    payload.factor_signal_path = signalPath;
                }
            } catch (err: unknown) {
                const details = err instanceof Error ? err.message : undefined;
                console.error('[Strategy API] Failed to create combined signal:', err);
                return NextResponse.json({
                    error: 'Failed to prepare factor signals',
                    details,
                    metrics: [],
                    daily_returns: []
                }, { status: 500 });
            }
        }

        if (!payload.pool_id && !payload.factor_signals && !payload.factor_signal_path) {
            return NextResponse.json({
                error: 'pool_id or factor signals are required',
                metrics: [],
                daily_returns: []
            }, { status: 400 });
        }

        console.log(`[Strategy API] Running lightweight computation${signalPath ? ` with cached signal ${path.basename(signalPath)}` : ''}`);

        const result = await runPythonScript(STRATEGY_SCRIPT, payload, STRATEGY_TIMEOUT_MS, '[Strategy]');

        if (result.error || !result.data) {
            return NextResponse.json({
                error: result.error || 'Strategy computation failed',
                details: result.details,
                metrics: [],
                daily_returns: []
            }, { status: 500 });
        }

        return NextResponse.json(result.data);

    } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Internal error';
        console.error('[Strategy API] Unexpected error', e);
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
