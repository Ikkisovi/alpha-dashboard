
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import zlib from 'zlib';
import { ComputeRequest, CacheEntry, ComputeResponse } from './types';

// Use /tmp for Vercel/Lambda compatibility
const CACHE_DIR = '/tmp/alphasage_cache';
const CACHE_VERSION = 'v2';

// Ensure cache directory exists
if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
}

class CacheManager {
    private memoryCache: Map<string, CacheEntry>;
    private maxMemorySize: number = 100 * 1024 * 1024; // 100MB
    private currentMemorySize: number = 0;
    private cacheDir: string = CACHE_DIR;

    constructor() {
        this.memoryCache = new Map();
    }

    generateKey(request: ComputeRequest): string {
        const parts = [
            CACHE_VERSION,
            request.pool_id,
            (request.factor_ids || []).sort().join(','),
            request.train_start,
            request.train_end,
            request.test_start,
            request.test_end,
            request.target_horizons.sort().join(','),
            request.strategy.type,
            request.strategy.top_pct,
            request.strategy.rebalance_days
        ];

        return crypto
            .createHash('sha256')
            .update(parts.join('|'))
            .digest('hex')
            .slice(0, 16);
    }

    get(key: string): ComputeResponse | null {
        const entry = this.memoryCache.get(key);
        if (!entry) return null;

        // Check TTL (5 minutes for memory)
        if (Date.now() - entry.timestamp > 5 * 60 * 1000) {
            this.evict(key);
            return null;
        }

        // Refresh LRU
        this.memoryCache.delete(key);
        this.memoryCache.set(key, entry);

        return entry.data;
    }

    async getFromFile(key: string): Promise<ComputeResponse | null> {
        const filePath = path.join(this.cacheDir, `${key}.json.gz`);
        if (!fs.existsSync(filePath)) return null;

        try {
            const stats = fs.statSync(filePath);
            // Check 24hr TTL for file
            if (Date.now() - stats.mtimeMs > 24 * 60 * 60 * 1000) {
                fs.unlinkSync(filePath);
                return null;
            }

            const compressed = fs.readFileSync(filePath);
            const json = zlib.gunzipSync(compressed).toString();
            const data = JSON.parse(json);

            return data;
        } catch (e) {
            console.error("Cache read error:", e);
            return null;
        }
    }

    set(key: string, data: ComputeResponse): void {
        const size = JSON.stringify(data).length;

        // Evict if full
        while (this.currentMemorySize + size > this.maxMemorySize && this.memoryCache.size > 0) {
            const firstKey = this.memoryCache.keys().next().value;
            if (firstKey) this.evict(firstKey);
        }

        this.memoryCache.set(key, {
            data,
            timestamp: Date.now(),
            size
        });
        this.currentMemorySize += size;
    }

    evict(key: string) {
        const entry = this.memoryCache.get(key);
        if (entry) {
            this.currentMemorySize -= entry.size;
            this.memoryCache.delete(key);
        }
    }

    async saveToFile(key: string, data: ComputeResponse): Promise<void> {
        try {
            const json = JSON.stringify(data);
            const compressed = zlib.gzipSync(json);

            fs.writeFileSync(
                path.join(this.cacheDir, `${key}.json.gz`),
                compressed
            );
        } catch (e) {
            console.error("Cache write error:", e);
        }
    }
}

export const cacheManager = new CacheManager();
