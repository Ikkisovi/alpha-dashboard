
import { ComputeRequest, ComputeResponse } from './types';

export interface StrategyRequest {
    pool_id?: string;
    factor_ids?: number[] | null;
    train_start?: string;
    train_end?: string;
    test_start?: string;
    test_end?: string;
    target_horizons?: number[];
    targetHorizons?: number[];
    strategy?: {
        type?: "long_short" | "long_only" | "equal_weight";
        top_pct?: number;
        rebalance_days?: number;
    };
    factor_signals?: number[][];
    factor_signal_path?: string;
}

export interface StrategyMetricResult {
    period: string;
    factor_id: number;
    ic: number;
    ic_std: number;
    icir: number;
    ric: number;
    annual_return: number;
    sharpe: number;
    max_drawdown: number;
}

export interface DailyReturnResult {
    date: string;
    factor_id: number;
    horizon_days?: number;
    return: number;
    benchmark?: number;
    rebalance_id?: number;
}

export interface StrategyResponse {
    metrics: StrategyMetricResult[];
    daily_returns: DailyReturnResult[];
    computation_time_ms: number;
    error?: string;
    details?: string;
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
            const message = error.details ? `${error.error}: ${error.details}` : error.error;
            throw new Error(message || `HTTP error ${response.status}`);
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
            const message = error.details ? `${error.error}: ${error.details}` : error.error;
            throw new Error(message || `HTTP error ${response.status}`);
        }

        return response.json();
    }
}
