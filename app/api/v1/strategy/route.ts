import { spawn } from 'child_process';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

interface StrategyRequest {
    train_start?: string;
    train_end?: string;
    test_start?: string;
    test_end?: string;
    strategy?: {
        type?: string;
        top_pct?: number;
        rebalance_days?: number;
    };
    factor_signals?: number[][];  // Optional pre-computed signals
}

export async function POST(request: NextRequest) {
    try {
        const body: StrategyRequest = await request.json();

        console.log('[Strategy API] Running lightweight strategy computation...');
        const result = await runStrategyComputation(body);

        if (result.error) {
            return NextResponse.json(result, { status: 500 });
        }

        return NextResponse.json(result);

    } catch (e: any) {
        return NextResponse.json({ error: e.message }, { status: 500 });
    }
}

async function runStrategyComputation(request: StrategyRequest): Promise<any> {
    return new Promise((resolve) => {
        // Resolve python path
        const venvPython = path.resolve(process.cwd(), '../.venv/bin/python');
        const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3';

        // Path to lightweight strategy script
        let scriptPath = path.resolve(process.cwd(), '../python-backend/strategy_service.py');
        if (!fs.existsSync(scriptPath)) {
            scriptPath = path.resolve(process.cwd(), 'python-backend/strategy_service.py');
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
                console.error("[Strategy] Python script failed:", stderr);
                resolve({
                    error: "Strategy computation failed",
                    details: stderr,
                    metrics: [],
                    daily_returns: []
                });
                return;
            }

            try {
                const json = JSON.parse(stdout);
                resolve(json);
            } catch (e) {
                console.error("[Strategy] Failed to parse output:", stdout);
                resolve({
                    error: "Invalid response from strategy backend",
                    details: stdout,
                    metrics: [],
                    daily_returns: []
                });
            }
        });

        // Write input
        child.stdin.write(JSON.stringify(request));
        child.stdin.end();

        // Short timeout - this should be fast (< 5s)
        setTimeout(() => {
            child.kill();
            resolve({ error: "Strategy computation timed out (15s limit)" });
        }, 15000);
    });
}
