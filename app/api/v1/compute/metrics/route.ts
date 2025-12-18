
import { spawn } from 'child_process';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { cacheManager } from '@/lib/cache-manager';
import { ComputeRequest } from '@/lib/types';

export async function POST(request: NextRequest) {
    try {
        const body: ComputeRequest = await request.json();

        // 1. Generate cache key
        const cacheKey = cacheManager.generateKey(body);

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

        // 4. Spawn Python computation
        console.log(`Cache miss for ${cacheKey}. Spawning Python...`);
        const result = await runPythonComputation(body);

        if (result.error) {
            return NextResponse.json(result, { status: 500 });
        }

        // 5. Store in cache (memory + file)
        cacheManager.set(cacheKey, result);
        await cacheManager.saveToFile(cacheKey, result);

        return NextResponse.json({
            ...result,
            cache_hit: false
        });

    } catch (e: any) {
        return NextResponse.json({ error: e.message }, { status: 500 });
    }
}

async function runPythonComputation(request: ComputeRequest): Promise<any> {
    return new Promise((resolve, reject) => {
        // Resolve python path: Use venv locally, python3 on Vercel
        // process.cwd() in Next.js dev is the dashboard directory
        const venvPython = path.resolve(process.cwd(), '../.venv/bin/python');
        const pythonCmd = require('fs').existsSync(venvPython) ? venvPython : 'python3';

        // Path to script (relative to dashboard directory)
        let scriptPath = path.resolve(process.cwd(), '../python-backend/compute_service.py');
        if (!require('fs').existsSync(scriptPath)) {
            // Fallback for Vercel deployment where python-backend is inside the project
            scriptPath = path.resolve(process.cwd(), 'python-backend/compute_service.py');
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
            } catch (e) {
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
            resolve({ error: "Computation timed out (60s limit)" });
        }, 60000);
    });
}
