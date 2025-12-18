import Papa from "papaparse";

export type StrategyMetric = {
    Dataset: string;
    ic: number;
    ic_std: number;
    icir: number;
    ric: number;
    ric_std: number;
    ricir: number;
    ret: number;
    ret_std: number;
    retir: number;
    ret_sharpe: number;
    ret_mdd: number;
};

export type DailyPnL = {
    date: string;
    daily_ret: number;
    benchmark?: number;
    cumulative_ret?: number;
    cumulative_benchmark?: number;
};

export type FactorPnL = {
    date: string;
    benchmark?: number;
    [key: string]: number | string | undefined;
};

export type HoldingPnL = DailyPnL & {
    rebalance_id: number;
};

export type FactorInfo = {
    factor_id: number;
    name: string;
    expr: string;
    description: string;
    type?: string;
    ic?: number;
    icir?: number;
    sharpe?: number;
    mdd?: number;
    source?: string;
};

export type CorrelationMatrix = {
    factors: string[];
    matrix: number[][];
};

export type VifScore = {
    factor_id: number;
    factor_name: string;
    vif: number;
    multicollinear: boolean;
};

export type GridSearchResult = {
    n_factors: number;
    r_squared: number;
    selected_factors: string;
    factor_names: string;
};

export async function loadCombinedFactors(): Promise<{ dictionary: FactorInfo[]; returns: (FactorPnL & { index: number })[] }> {
    const response = await fetch("/api/v1/custom-factors");
    if (!response.ok) {
        throw new Error(`Failed to load combined factors: ${response.status}`);
    }
    const data = await response.json();
    const dict = (data.dictionary || []) as FactorInfo[];
    const returns = (data.returns || []) as (FactorPnL & { index: number })[];
    return { dictionary: dict, returns };
}

export async function loadStrategyMetrics(): Promise<StrategyMetric[]> {
    const response = await fetch("/data/strategy/metrics.csv");
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                const data = results.data.map((row: any) => {
                    const dataset = row[""] || row["Dataset"] || row[Object.keys(row)[0]];
                    return { ...row, Dataset: dataset };
                });
                resolve(data as StrategyMetric[]);
            },
        });
    });
}

export async function loadStrategyDailyPnL(): Promise<(DailyPnL & { index: number })[]> {
    const response = await fetch("/data/strategy/daily_pnl.csv");
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
                const data = results.data as DailyPnL[];
                let cumulative = 0;
                let cumulativeBenchmark = 0;
                const processed = data.map((d, i) => {
                    if (typeof d.daily_ret === 'number') cumulative += d.daily_ret;
                    if (typeof d.benchmark === 'number') cumulativeBenchmark += d.benchmark;
                    return {
                        ...d,
                        cumulative_ret: cumulative,
                        cumulative_benchmark: cumulativeBenchmark,
                        index: i
                    };
                }).filter(d => typeof d.daily_ret === 'number');
                resolve(processed as (DailyPnL & { index: number })[]);
            },
        });
    });
}

export async function loadHoldingsDailyPnL(): Promise<(HoldingPnL & { index: number })[]> {
    try {
        const response = await fetch("/data/strategy_daily_holdings_pnl.csv");
        if (!response.ok) return [];
        const csvText = await response.text();
        return new Promise((resolve) => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data as HoldingPnL[];
                    let cumulative = 0;
                    let cumulativeBenchmark = 0;
                    const processed = data
                        .filter(d => typeof d.daily_ret === 'number')
                        .map((d, i) => {
                            cumulative += d.daily_ret || 0;
                            cumulativeBenchmark += d.benchmark || 0;
                            return {
                                ...d,
                                cumulative_ret: cumulative,
                                cumulative_benchmark: cumulativeBenchmark,
                                index: i
                            };
                        });
                    resolve(processed as (HoldingPnL & { index: number })[]);
                },
            });
        });
    } catch { return []; }
}

export async function loadFactorReturns(): Promise<(FactorPnL & { index: number })[]> {
    const response = await fetch("/data/factors/returns.csv");
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                const data = results.data as FactorPnL[];
                const withIndex = data
                    .filter(d => Object.keys(d).length > 0)
                    .map((d, i) => ({ ...d, index: i }));
                resolve(withIndex);
            },
        });
    });
}

// Alias for compatibility if needed, or replace usages
export const loadIndividualFactorsPnL = loadFactorReturns;

// Helper function to infer factor type from name/expression
function inferFactorType(name: string, expr: string): string {
    const lowerExpr = (expr || '').toLowerCase();
    const lowerName = (name || '').toLowerCase();

    if (lowerExpr.includes('tsret') || lowerExpr.includes('tsmomrank') || lowerExpr.includes('ret'))
        return 'momentum';
    if (lowerExpr.includes('volume') || lowerName.includes('volume'))
        return 'volume';
    if (lowerExpr.includes('tsskew') || lowerExpr.includes('tsvar') || lowerExpr.includes('tsstd'))
        return 'volatility';
    if (lowerExpr.includes('rank') || lowerExpr.includes('quantile'))
        return 'mean_reversion';
    if (lowerExpr.includes('tscorr') || lowerExpr.includes('corr'))
        return 'correlation';
    return 'composite';
}

export async function loadFactorDictionary(): Promise<FactorInfo[]> {
    const response = await fetch("/data/factors/dictionary.csv");
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                const data = results.data.map((row: any) => ({
                    ...row,
                    expr: row.expression,  // Map expression to expr
                    type: inferFactorType(row.name, row.expression),  // Derive type
                }));
                resolve(data as FactorInfo[]);
            },
        });
    });
}

export async function loadFactorMetrics(): Promise<any[]> {
    const response = await fetch("/data/factors/metrics.csv");
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                resolve(results.data as any[]);
            },
        });
    });
}

export async function loadCorrelationMatrix(): Promise<CorrelationMatrix | null> {
    try {
        const response = await fetch("/data/factors/correlation.csv");
        if (!response.ok) return null;
        const csvText = await response.text();
        return new Promise((resolve) => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data as Record<string, number | string>[];
                    if (data.length === 0) { resolve(null); return; }
                    const factors = data.map(row => String(row[""] || row[Object.keys(row)[0]]));
                    const matrix = data.map(row => {
                        const values: number[] = [];
                        Object.keys(row).forEach((key, idx) => {
                            if (idx > 0) values.push(Number(row[key]) || 0);
                        });
                        return values;
                    });
                    resolve({ factors, matrix });
                },
            });
        });
    } catch { return null; }
}

export async function loadVifScores(): Promise<VifScore[]> {
    try {
        const response = await fetch("/data/factor_vif_scores.csv");
        if (!response.ok) return [];
        const csvText = await response.text();
        return new Promise((resolve) => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data.map((row: any) => ({
                        factor_id: row.factor_id,
                        factor_name: row.factor_name,
                        vif: row.vif,
                        multicollinear: String(row.multicollinear).toLowerCase() === 'true',
                    }));
                    resolve(data as VifScore[]);
                },
            });
        });
    } catch { return []; }
}

export async function loadGridSearchResults(): Promise<GridSearchResult[]> {
    try {
        const response = await fetch("/data/factor_grid_search.csv");
        if (!response.ok) return [];
        const csvText = await response.text();
        return new Promise((resolve) => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    resolve(results.data as GridSearchResult[]);
                },
            });
        });
    } catch { return []; }
}
