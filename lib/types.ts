
export interface ComputeRequest {
    pool_id: string;
    factor_ids?: number[] | null;
    train_start: string;
    train_end: string;
    test_start: string;
    test_end: string;
    target_horizons: number[];
    skip_analytics?: boolean;
    export_factor_values_csv?: string;
    auto_increment_factor_ids?: boolean;
    factor_id_offset?: number;
    strategy: {
        type: "long_short" | "long_only" | "equal_weight";
        top_pct: number;
        rebalance_days: number;
    };
}

export interface Metric {
    factor_id: number;
    factor_name: string;
    horizon_days: number;
    period: "train" | "test";
    ic: number;
    ic_std: number;
    icir: number;
    ric: number;
    sharpe: number;
    annual_return: number;
    daily_return_mean: number;
    max_drawdown: number;
}

export interface DailyReturn {
    date: string; // YYYY-MM-DD
    factor_id: number;
    horizon_days: number;
    return: number;
    benchmark?: number;
    rebalance_id?: number;
}

export interface ComputeResponse {
    metrics: Metric[];
    daily_returns: DailyReturn[];
    correlation_matrix?: number[][];
    vif_scores?: ComputeVifScore[];
    rolling_ic?: RollingIC[];
    grid_search_results?: ComputeGridSearchResult[];
    factor_dictionary?: ComputeFactorInfo[];
    computation_time_ms: number;
    cache_hit?: boolean;
    parse_warnings?: string[];
    pool_path?: string;
    factor_values_csv?: string;
    factor_id_offset?: number;
    error?: string;
    details?: string;
    traceback?: string;
}

export interface CacheEntry {
    data: ComputeResponse;
    timestamp: number;
    size: number;
}

export interface RollingIC {
    date: string;
    factor_id: number;
    ic: number;
}

export interface ComputeVifScore {
    factor_id: number;
    vif: number;
    multicollinear: boolean;
}

export interface ComputeGridSearchResult {
    n_factors: number;
    r_squared: number;
    selected_factors: number[];
}

export interface ComputeFactorInfo {
    factor_id: number;
    name: string;
    expression: string;
    type: string;
    description: string;
    ic: number;
    icir: number;
    sharpe: number;
    mdd: number;
}
