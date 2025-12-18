
import { ComputeRequest, ComputeResponse } from './types';

export interface StrategyRequest {
    train_start?: string;
    train_end?: string;
    test_start?: string;
    test_end?: string;
    strategy?: {
        type?: string;
        top_pct?: number;
        rebalance_days?: number;
    };
    factor_signals?: number[][];
}

export interface StrategyResponse {
    metrics: any[];
    daily_returns: any[];
    computation_time_ms: number;
    error?: string;
}

export class ApiClient {
    private static baseUrl = '/api/v1';

    static async computeMetrics(request: ComputeRequest): Promise<ComputeResponse> {
        const response = await fetch(`${this.baseUrl}/compute/metrics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP error ${response.status}`);
        }

        return response.json();
    }

    /**
     * Lightweight strategy-only computation (much faster than computeMetrics)
     * Use this when only strategy parameters change (top_pct, rebalance_days, type)
     */
    static async computeStrategy(request: StrategyRequest): Promise<StrategyResponse> {
        const response = await fetch(`${this.baseUrl}/strategy`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP error ${response.status}`);
        }

        return response.json();
    }
}
