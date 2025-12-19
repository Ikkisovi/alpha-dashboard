"use client";

import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import {
  loadStrategyDailyPnL,
  loadStrategyMetrics,
  loadCombinedFactors,
  loadIndividualFactorsPnL,
  loadFactorMetrics,
  loadCorrelationMatrix,
  loadVifScores,
  loadGridSearchResults,
  loadHoldingsDailyPnL,
  StrategyMetric,
  DailyPnL,
  HoldingPnL,
  FactorPnL,
  FactorInfo,
  CorrelationMatrix,
  VifScore,
  GridSearchResult,
} from "@/utils/data";
import { RollingIC } from "@/lib/types";
import { TrendingUp, Activity, BarChart3, Search, SquareTerminal, AlertTriangle, Grid3X3, BookOpen, Info, LucideIcon, Download } from "lucide-react";

// Bloomberg-style Card Wrapper
function BBCard({ children, className, title, icon: Icon, span = "col-span-12" }: { children: React.ReactNode, className?: string, title?: string, icon?: LucideIcon, span?: string }) {
  return (
    <div className={`${span} border border-gray-300 dark:border-gray-800 bg-white dark:bg-black p-0 flex flex-col ${className}`}>
      {title && (
        <div className="border-b border-gray-300 dark:border-gray-800 px-3 py-2 bg-gray-50 dark:bg-[#111] flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-900 dark:text-amber-500 flex items-center gap-2">
            {Icon && <Icon className="h-3 w-3" />}
            {title}
          </h3>
        </div>
      )}
      <div className="p-3">
        {children}
      </div>
    </div>
  );
}

function MetricValue({ label, value, sub, color = "text-gray-900 dark:text-gray-100" }: { label: string, value: string, sub?: string, color?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase text-gray-500 font-semibold tracking-wider">{label}</span>
      <span className={`text-xl font-mono font-bold ${color}`}>{value}</span>
      {sub && <span className="text-[10px] text-gray-400">{sub}</span>}
    </div>
  );
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<StrategyMetric[]>([]);
  const [strategyPnL, setStrategyPnL] = useState<(DailyPnL & { index: number })[]>([]);
  const [holdingsPnL, setHoldingsPnL] = useState<(HoldingPnL & { index: number, cumulative_ret?: number, cumulative_benchmark?: number })[]>([]);
  const [factorPnL, setFactorPnL] = useState<(FactorPnL & { index: number })[]>([]);
  const [cumulativeFactorPnL, setCumulativeFactorPnL] = useState<(FactorPnL & { index: number })[]>([]);
  const [selectedFactors, setSelectedFactors] = useState<string[]>([]);
  const [backtestMode, setBacktestMode] = useState<'all' | 'selected'>('all');
  const [loading, setLoading] = useState(true);

  // Factor dictionary and detail panel
  const [factorDictionary, setFactorDictionary] = useState<FactorInfo[]>([]);
  const [selectedFactorDetail, setSelectedFactorDetail] = useState<FactorInfo | null>(null);
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const [lastPoolLoaded, setLastPoolLoaded] = useState<string | null>(null);
  const [factorDatasetReady, setFactorDatasetReady] = useState(false);
  const [isFactorLoading, setIsFactorLoading] = useState(false);
  const [factorLoadError, setFactorLoadError] = useState<string | null>(null);
  const [perStockFactorCsvUrl, setPerStockFactorCsvUrl] = useState<string | null>(null);
  const [minDate, setMinDate] = useState<string>("");
  const [maxDate, setMaxDate] = useState<string>("");

  // Date-based range (YYYY-MM-DD format)
  const [dateRange, setDateRange] = useState<{ startDate: string; endDate: string }>({ startDate: '', endDate: '' });
  const [showCumulative, setShowCumulative] = useState(true);
  const [allStrategyPnL, setAllStrategyPnL] = useState<(DailyPnL & { index: number })[]>([]);
  const [allHoldingsPnL, setAllHoldingsPnL] = useState<(HoldingPnL & { index: number, cumulative_ret?: number, cumulative_benchmark?: number })[]>([]);
  const [allFactorPnL, setAllFactorPnL] = useState<(FactorPnL & { index: number })[]>([]);
  const [allCumulativeFactorPnL, setAllCumulativeFactorPnL] = useState<(FactorPnL & { index: number })[]>([]);
  const [selectedRebalance, setSelectedRebalance] = useState<number | "all">("all");

  /* New State for Dynamic Computation */
  const [computeConfig, setComputeConfig] = useState({
    poolId: '/Users/ikki/AlphaSAGE/data/ppo_logs/pool_30/ppo_legacy_30_0-20251212204441/ppo_legacy_30_0_20251212204441/200704_steps_pool.json',
    targetHorizons: [5],
    strategy: {
      type: 'long_short',
      topPct: 0.2,
      rebalanceDays: 5
    }
  });
  const [isComputing, setIsComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);

  // Sorting State
  const [sortConfig, setSortConfig] = useState<{ key: string, direction: 'asc' | 'desc' }>({ key: 'ic', direction: 'desc' });

  // Correlation, VIF, and Grid Search data
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationMatrix | null>(null);
  const [vifScores, setVifScores] = useState<VifScore[]>([]);
  const [gridSearchResults, setGridSearchResults] = useState<GridSearchResult[]>([]);
  const [rollingIC, setRollingIC] = useState<RollingIC[]>([]);
  const [factorViewMode, setFactorViewMode] = useState<'returns' | 'rolling_ic'>('returns');
  const rebalanceOptions = Array.from(
    new Set(allHoldingsPnL.map(d => d.rebalance_id).filter((v): v is number => typeof v === 'number'))
  ).sort((a, b) => a - b);

  // Data Loading Implementation
  const fetchFactorDataset = async () => {
    setIsFactorLoading(true);
    setFactorLoadError(null);
    setFactorDatasetReady(false);
    try {
      const poolLabel = computeConfig.poolId.split('/').pop() || 'pool';
      const safePoolLabel = poolLabel.replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '') || 'pool';
      const exportFile = `public/data/factor_values/stock_factor_values_${safePoolLabel}.csv`;
      const req = {
        pool_id: computeConfig.poolId,
        factor_ids: null,
        train_start: dateRange.startDate || '2022-01-01',
        train_end: '2023-12-31',
        test_start: '2024-01-01',
        test_end: dateRange.endDate || '2024-12-31',
        target_horizons: computeConfig.targetHorizons,
        export_factor_values_csv: exportFile,
        auto_increment_factor_ids: true,
        strategy: {
          type: computeConfig.strategy.type as 'long_short' | 'long_only' | 'equal_weight',
          top_pct: computeConfig.strategy.topPct,
          rebalance_days: computeConfig.strategy.rebalanceDays
        }
      };

      const data = await import('@/lib/api-client').then(m => m.ApiClient.computeMetrics(req));
      if (data.error) throw new Error(data.error);
      setLastPoolLoaded(data.pool_path || computeConfig.poolId);
      const rawCsvPath = data.factor_values_csv || exportFile;
      if (rawCsvPath) {
        const normalized = rawCsvPath.replace(/\\/g, "/");
        const publicIndex = normalized.indexOf("/public/");
        let url = normalized;
        if (publicIndex >= 0) {
          url = normalized.slice(publicIndex + "/public".length);
        } else if (normalized.startsWith("public/")) {
          url = normalized.slice("public".length);
        }
        if (!url.startsWith("/")) url = `/${url}`;
        setPerStockFactorCsvUrl(url);
      }

      const dateMap = new Map<string, Record<string, unknown>>();
      data.daily_returns.forEach((r) => {
        if (r.factor_id === -1) return;
        if (!dateMap.has(r.date)) dateMap.set(r.date, { date: r.date, index: 0 });
        const entry = dateMap.get(r.date)!;
        entry[`factor_${r.factor_id}`] = r.return;
      });
      const newFactorPnL = Array.from(dateMap.values())
        .sort((a, b) => String(a.date).localeCompare(String(b.date)))
        .map((d, i) => ({ ...d, index: i })) as (FactorPnL & { index: number })[];
      setFactorPnL(newFactorPnL);
      setAllFactorPnL(newFactorPnL);

      const cumulatives: (FactorPnL & { index: number })[] = [];
      const runningTotals: { [key: string]: number } = {};
      const factorKeys = newFactorPnL.length > 0 ? Object.keys(newFactorPnL[0]).filter(k => k !== 'index' && k !== 'date') : [];
      factorKeys.forEach(k => { runningTotals[k] = 0.0; });

      newFactorPnL.forEach((row, idx) => {
        const newRow: FactorPnL & { index: number } = { index: idx, date: row.date };
        const rowRecord = row as Record<string, unknown>;
        Object.keys(rowRecord).forEach(k => {
          if (k !== 'index' && k !== 'date' && typeof rowRecord[k] === 'number') {
            runningTotals[k] = (runningTotals[k] || 0.0) + (rowRecord[k] as number);
            newRow[k] = (runningTotals[k]) * 100;
          } else if (typeof rowRecord[k] === 'string') {
            newRow[k] = rowRecord[k] as string;
          }
        });
        cumulatives.push(newRow);
      });
      setCumulativeFactorPnL(cumulatives);
      setAllCumulativeFactorPnL(cumulatives);

      const availableFactors = newFactorPnL.length > 0 ? Object.keys(newFactorPnL[0])
        .filter(k => k.startsWith('factor_'))
        .map(k => parseInt(k.replace('factor_', ''), 10))
        : [];

      if (data.factor_dictionary && data.factor_dictionary.length > 0) {
        const mapped = data.factor_dictionary.map((d) => ({ ...d, expr: d.expression })) as FactorInfo[];
        const dedupedMap = new Map<number, FactorInfo>();
        mapped.forEach(f => { if (!dedupedMap.has(f.factor_id)) dedupedMap.set(f.factor_id, f); });
        const deduped = Array.from(dedupedMap.values());
        setFactorDictionary(deduped);
      } else if (availableFactors.length > 0) {
        setFactorDictionary(availableFactors.map(id => ({
          factor_id: id,
          name: `Factor ${id}`,
          expr: 'Derived from backend returns',
          description: '',
          type: 'composite',
          ic: 0, icir: 0, sharpe: 0, mdd: 0,
          source: data.pool_path || computeConfig.poolId
        })));
      }

      if (newFactorPnL.length > 0) {
        const factorKeys = Object.keys(newFactorPnL[0]).filter(k => k.startsWith('factor_'));
        const validPrevious = selectedFactors.filter(f => factorKeys.includes(f));
        if (validPrevious.length > 0) setSelectedFactors(validPrevious);
        else setSelectedFactors(factorKeys.slice(0, 20));

        const dynamicDates = [newFactorPnL[0].date, newFactorPnL[newFactorPnL.length - 1].date].filter(Boolean).sort();
        if (dynamicDates.length >= 2) {
          setDateRange({ startDate: dynamicDates[0], endDate: dynamicDates[dynamicDates.length - 1] });
          setMinDate(dynamicDates[0]);
          setMaxDate(dynamicDates[dynamicDates.length - 1]);
        }
      }
      setFactorDatasetReady(true);
    } catch (e: unknown) {
      console.error(e);
      setFactorLoadError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setIsFactorLoading(false);
    }
  };

  const fetchDynamicMetrics = async () => {
    setIsComputing(true);
    setComputeError(null);
    setMetrics([]);
    setStrategyPnL([]);
    setHoldingsPnL([]);
    // Keep analytics from previous load (don't clear them for strategy-only changes)
    // setRollingIC([]);
    // setCorrelationMatrix(null);
    // setVifScores([]);
    // setGridSearchResults([]);
    try {
      const selectedFactorIds = backtestMode === 'selected'
        ? selectedFactors
          .map((f) => Number.parseInt(f.replace('factor_', ''), 10))
          .filter((id) => Number.isFinite(id))
        : [];

      const req = {
        pool_id: computeConfig.poolId,
        factor_ids: (backtestMode === 'selected' && selectedFactorIds.length > 0) ? selectedFactorIds : null,
        train_start: dateRange.startDate || '2022-01-01',
        train_end: '2023-12-31',
        test_start: '2024-01-01',
        test_end: dateRange.endDate || '2024-12-31',
        target_horizons: computeConfig.targetHorizons,
        strategy: {
          type: computeConfig.strategy.type as 'long_short' | 'long_only' | 'equal_weight',
          top_pct: computeConfig.strategy.topPct,
          rebalance_days: computeConfig.strategy.rebalanceDays
        }
      };

      // Compute strategy quickly using cached combined signals.
      const data = await import('@/lib/api-client').then(m => m.ApiClient.computeStrategy(req));
      if (data.error) throw new Error(data.error);
      console.log(`[Backtest] Computed in ${data.computation_time_ms?.toFixed(0) || '?'}ms`);

      const newMetrics: StrategyMetric[] = data.metrics.map(m => ({
        Dataset: m.period === 'train' ? 'Validation' : 'Test',
        ic: m.ic, ic_std: m.ic_std, icir: m.icir, ric: m.ric, ric_std: 0, ricir: 0,
        ret: m.annual_return, ret_std: 0, retir: 0, ret_sharpe: m.sharpe, ret_mdd: m.max_drawdown
      }));
      const stratMetrics = newMetrics.filter((_, i) => data.metrics[i].factor_id === -1);
      setMetrics(stratMetrics.length > 0 ? stratMetrics : newMetrics);

      const primaryHorizon = computeConfig.targetHorizons?.[0];
      const stratReturns = data.daily_returns.filter((r) =>
        r.factor_id === -1 && (primaryHorizon ? r.horizon_days === primaryHorizon : true)
      );

      let computedStrategy: (DailyPnL & { index: number })[] = [];
      let computedHoldings: (HoldingPnL & { index: number, cumulative_ret?: number, cumulative_benchmark?: number })[] = [];

      if (stratReturns.length > 0 || data.daily_returns.length > 0) {
        const fallbackFactorId = backtestMode === 'selected' && selectedFactors.length > 0
          ? Number.parseInt(selectedFactors[0].replace('factor_', ''), 10)
          : data.daily_returns.find((r) => r.factor_id !== -1)?.factor_id;
        const fallbackReturns = (typeof fallbackFactorId === 'number' && !Number.isNaN(fallbackFactorId))
          ? data.daily_returns.filter((r) => r.factor_id === fallbackFactorId && (primaryHorizon ? r.horizon_days === primaryHorizon : true))
          : [];
        const source = (stratReturns.length > 0 ? stratReturns : fallbackReturns)
          .slice()
          .sort((a, b) => a.date.localeCompare(b.date));

        let strategyEquity = 1.0;
        let benchmarkEquity = 1.0;
        const newStratPnL = source.map((r, i) => {
          const dailyRet = typeof r.return === 'number' ? r.return : 0;
          const benchRet = typeof r.benchmark === 'number' ? r.benchmark : 0;
          strategyEquity *= (1.0 + dailyRet);
          benchmarkEquity *= (1.0 + benchRet);
          return {
            date: r.date,
            daily_ret: dailyRet,
            benchmark: benchRet,
            cumulative_ret: (strategyEquity - 1.0) * 100,
            cumulative_benchmark: (benchmarkEquity - 1.0) * 100,
            index: i
          };
        });
        computedStrategy = newStratPnL;
        setStrategyPnL(newStratPnL);
        setAllStrategyPnL(newStratPnL);

        const rebalanceDays = computeConfig.strategy.rebalanceDays || 5;
        const newHoldingsPnL = newStratPnL.map((r, i) => ({
          ...r,
          rebalance_id: source[i].rebalance_id ?? Math.floor(i / rebalanceDays),
        }));
        computedHoldings = newHoldingsPnL;
        setHoldingsPnL(newHoldingsPnL);
        setAllHoldingsPnL(newHoldingsPnL);
        setSelectedRebalance("all");
      }

      const dynamicDates = [];
      if (computedStrategy.length > 0) dynamicDates.push(computedStrategy[0].date, computedStrategy[computedStrategy.length - 1].date);
      if (computedHoldings.length > 0) dynamicDates.push(computedHoldings[0].date, computedHoldings[computedHoldings.length - 1].date);
      const trimmedDates = dynamicDates.filter(Boolean).sort();
      if (trimmedDates.length >= 2) {
        setDateRange({ startDate: trimmedDates[0], endDate: trimmedDates[trimmedDates.length - 1] });
        setMinDate(trimmedDates[0]);
        setMaxDate(trimmedDates[trimmedDates.length - 1]);
      }
    } catch (e: unknown) {
      console.error(e);
      setComputeError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setIsComputing(false);
    }
  };

  useEffect(() => {
    async function fetchData() {
      const [m, s, combined, fLegacy, fMetrics, corrMatrix, vif, gridSearch, holdings] = await Promise.all([
        loadStrategyMetrics(), loadStrategyDailyPnL(), loadCombinedFactors(), loadIndividualFactorsPnL(),
        loadFactorMetrics(), loadCorrelationMatrix(), loadVifScores(), loadGridSearchResults(), loadHoldingsDailyPnL(),
      ]);
      setMetrics(m);
      const fd = combined.dictionary;
      const factorReturns = (combined.returns && combined.returns.length > 0) ? combined.returns : fLegacy;
      const enrichedDictionary = fd.map(factor => ({ ...factor, ...fMetrics.find((fm) => (fm as { factor_id?: number }).factor_id === factor.factor_id) }));
      setFactorDictionary(enrichedDictionary);
      setCorrelationMatrix(corrMatrix);
      setVifScores(vif);
      setGridSearchResults(gridSearch);

      setAllHoldingsPnL(holdings);
      setHoldingsPnL(holdings);
      setAllStrategyPnL(s as (DailyPnL & { index: number })[]);
      setStrategyPnL(s as (DailyPnL & { index: number })[]);

      if (factorReturns.length > 0) {
        const factorKeys = Object.keys(factorReturns[0]);
        const cumulatives: (FactorPnL & { index: number })[] = [];
        const runningTotals: { [key: string]: number } = {};
        factorKeys.forEach(k => runningTotals[k] = 0);
        factorReturns.forEach(row => {
          const newRow: FactorPnL & { index: number } = { index: row.index, date: row.date };
          factorKeys.forEach(k => {
            if (k !== 'index' && k !== 'date' && typeof row[k] === 'number') {
              runningTotals[k] = (runningTotals[k]) + (row[k] as number);
              newRow[k] = runningTotals[k] * 100;
            } else if (typeof row[k] === 'string') newRow[k] = row[k];
          });
          cumulatives.push(newRow);
        });
        setAllFactorPnL(factorReturns as (FactorPnL & { index: number })[]);
        setAllCumulativeFactorPnL(cumulatives);
        setFactorPnL(factorReturns as (FactorPnL & { index: number })[]);
        setCumulativeFactorPnL(cumulatives);
      }

      const allDates = [];
      if (s.length > 0) allDates.push(s[0].date, s[s.length - 1].date);
      if (factorReturns.length > 0) allDates.push(factorReturns[0].date, factorReturns[factorReturns.length - 1].date);
      if (holdings.length > 0) allDates.push(holdings[0].date, holdings[holdings.length - 1].date);
      const validDates = allDates.filter(d => d).sort();
      if (validDates.length > 0) {
        setDateRange({ startDate: validDates[0], endDate: validDates[validDates.length - 1] });
        setMinDate(validDates[0]);
        setMaxDate(validDates[validDates.length - 1]);
      }
      setLoading(false);
      if (factorReturns.length > 0) setFactorDatasetReady(true);
    }
    fetchData();
  }, []);

  useEffect(() => {
    if (allStrategyPnL.length === 0) return;
    let filteredStrategy = allStrategyPnL;
    let filteredFactorDaily = allFactorPnL;
    let filteredFactorCumulative = allCumulativeFactorPnL;
    let filteredHoldings = allHoldingsPnL;

    if (dateRange.startDate) {
      filteredStrategy = filteredStrategy.filter(d => d.date && d.date >= dateRange.startDate);
      filteredFactorDaily = filteredFactorDaily.filter(d => d.date && d.date >= dateRange.startDate);
      filteredFactorCumulative = filteredFactorCumulative.filter(d => d.date && d.date >= dateRange.startDate);
      filteredHoldings = filteredHoldings.filter(d => d.date && d.date >= dateRange.startDate);
    }
    if (dateRange.endDate) {
      filteredStrategy = filteredStrategy.filter(d => d.date && d.date <= dateRange.endDate);
      filteredFactorDaily = filteredFactorDaily.filter(d => d.date && d.date <= dateRange.endDate);
      filteredFactorCumulative = filteredFactorCumulative.filter(d => d.date && d.date <= dateRange.endDate);
      filteredHoldings = filteredHoldings.filter(d => d.date && d.date <= dateRange.endDate);
    }
    if (selectedRebalance !== "all") {
      filteredHoldings = filteredHoldings.filter(d => d.rebalance_id === selectedRebalance);
    }

    let stratEquity = 1.0;
    let stratBenchmarkEquity = 1.0;
    const strategyWithCompounded = filteredStrategy.map((d) => {
      stratEquity *= (1.0 + (d.daily_ret || 0));
      stratBenchmarkEquity *= (1.0 + (d.benchmark || 0));
      return { ...d, cumulative_ret: (stratEquity - 1.0) * 100, cumulative_benchmark: (stratBenchmarkEquity - 1.0) * 100 };
    });

    let holdingsEquity = 1.0;
    let holdingsBenchmarkEquity = 1.0;
    const holdingsWithCompounded = filteredHoldings.map((d) => {
      holdingsEquity *= (1.0 + (d.daily_ret || 0));
      holdingsBenchmarkEquity *= (1.0 + (d.benchmark || 0));
      return { ...d, cumulative_ret: (holdingsEquity - 1.0) * 100, cumulative_benchmark: (holdingsBenchmarkEquity - 1.0) * 100 };
    });

    setStrategyPnL(strategyWithCompounded);
    setFactorPnL(filteredFactorDaily);
    setCumulativeFactorPnL(filteredFactorCumulative);
    setHoldingsPnL(holdingsWithCompounded);
  }, [dateRange, allStrategyPnL, allHoldingsPnL, allFactorPnL, allCumulativeFactorPnL, selectedRebalance]);

  const toggleFactor = (factor: string) => {
    if (selectedFactors.includes(factor)) setSelectedFactors(selectedFactors.filter((f) => f !== factor));
    else setSelectedFactors([...selectedFactors, factor]);
  };

  const exportRawFactorValues = () => {
    if (perStockFactorCsvUrl) {
      const link = document.createElement("a");
      const fileName = perStockFactorCsvUrl.split("/").pop() || "stock_factor_values.csv";
      link.href = `/api/v1/factor-values?file=${encodeURIComponent(fileName)}`;
      link.download = fileName;
      link.click();
      return;
    }
    if (factorPnL.length === 0) return;
    const headers = ["date", ...Object.keys(factorPnL[0] || {}).filter(k => k !== "date" && k !== "index")];
    const csvContent = [
      headers.join(","),
      ...factorPnL.map(row => headers.map(h => row[h] ?? "").join(","))
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `factor_values_${Date.now()}.csv`;
    link.click();
  };

  const exportFactorDictionary = () => {
    if (factorDictionary.length === 0) return;
    const blob = new Blob([JSON.stringify(factorDictionary, null, 2)], { type: "application/json;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `factor_dictionary_${Date.now()}.json`;
    link.click();
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="font-mono text-amber-500 animate-pulse">LOADING ALPHABOARD...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black text-gray-900 dark:text-gray-100 font-sans p-4">
      <div className="grid grid-cols-12 gap-4">

        {/* Header */}
        <div className="col-span-12 flex items-center justify-between border-b-2 border-amber-600 pb-2 mb-4">
          <div className="flex items-center gap-2">
            <SquareTerminal className="h-6 w-6 text-amber-600" />
            <h1 className="text-2xl font-bold tracking-tighter uppercase dark:text-gray-100">AlphaBoard <span className="text-amber-600">PRO</span></h1>
          </div>
          <div className="font-mono text-xs text-right hidden md:block">
            <div className="text-gray-500">MARKET STATUS: OPEN</div>
            <div className="text-amber-600">{new Date().toISOString().split('T')[0]} | {new Date().toLocaleTimeString()}</div>
          </div>
        </div>

        {/* --- Top Row: Strategy Config & Controls --- */}
        {/* Factor Loading */}
        <BBCard title="DATA FEEDS & SOURCES" icon={Activity} span="col-span-12 lg:col-span-4">
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] font-bold uppercase text-gray-500">Source Pool</label>
              <select
                className="w-full bg-gray-100 dark:bg-[#1a1a1a] border border-gray-300 dark:border-gray-700 text-xs p-2 font-mono"
                value={computeConfig.poolId}
                onChange={e => { setFactorDatasetReady(false); setComputeConfig({ ...computeConfig, poolId: e.target.value }); }}
              >
                <option value="/Users/ikki/AlphaSAGE/data/ppo_logs/pool_30/ppo_legacy_30_0-20251212204441/ppo_legacy_30_0_20251212204441/200704_steps_pool.json">PPO Legacy 30 (200k)</option>
                <option value="/Users/ikki/AlphaSAGE/data/gfn_logs/pool_30/gfn_gnn_legacy_30_0_v24/pool_50000.json">GFN Legacy v24 (50k)</option>
              </select>
            </div>
            <button
              onClick={fetchFactorDataset}
              disabled={isFactorLoading}
              className="w-full py-2 bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold uppercase tracking-wider disabled:opacity-50"
            >
              {isFactorLoading ? 'Pulling Data...' : 'LOAD FACTOR DATA'}
            </button>
            <div className="flex justify-between items-center text-[10px] font-mono">
              <span className={factorDatasetReady ? "text-green-500" : "text-red-500"}>STATUS: {factorDatasetReady ? "READY" : "NO DATA"}</span>
              {lastPoolLoaded && <span className="text-gray-500 truncate w-32" title={lastPoolLoaded}>{lastPoolLoaded.split('/').pop()}</span>}
            </div>
          </div>
        </BBCard>

        {/* Strategy Parameters */}
        <BBCard title="STRATEGY CONFIGURATION" icon={Search} span="col-span-12 lg:col-span-8">
          <div className="flex flex-col gap-4">
            {/* Row 1: Parameters */}
            <div className="flex flex-col md:flex-row gap-4 items-end">
              <div className="flex-1 space-y-1 w-full">
                <label className="text-[10px] font-bold uppercase text-gray-500">Lookahead</label>
                <div className="flex gap-1">
                  {[5, 10, 20].map(d => (
                    <button
                      key={d}
                      onClick={() => setComputeConfig({ ...computeConfig, targetHorizons: [d], strategy: { ...computeConfig.strategy, rebalanceDays: d } })}
                      className={`flex-1 py-1 text-xs font-mono border ${computeConfig.targetHorizons.includes(d) ? 'bg-amber-600 border-amber-600 text-white' : 'bg-transparent border-gray-600 text-gray-500'}`}
                    >
                      {d}D
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex-1 space-y-1 w-full">
                <label className="text-[10px] font-bold uppercase text-gray-500">Direction</label>
                <select
                  className="w-full bg-gray-100 dark:bg-[#1a1a1a] border border-gray-300 dark:border-gray-700 text-xs p-1.5 font-mono"
                  value={computeConfig.strategy.type}
                  onChange={e => setComputeConfig({ ...computeConfig, strategy: { ...computeConfig.strategy, type: e.target.value } })}
                >
                  <option value="long_short">LONG/SHORT</option>
                  <option value="long_only">LONG ONLY</option>
                </select>
              </div>
              <div className="flex-1 space-y-1 w-full">
                <label className="text-[10px] font-bold uppercase text-gray-500">Top/Bottom %</label>
                <input
                  type="number" className="w-full bg-gray-100 dark:bg-[#1a1a1a] border border-gray-300 dark:border-gray-700 text-xs p-1.5 font-mono"
                  min={1} max={50} value={Math.round(computeConfig.strategy.topPct * 100)}
                  onChange={(e) => setComputeConfig({ ...computeConfig, strategy: { ...computeConfig.strategy, topPct: Math.max(1, Number(e.target.value)) / 100 } })}
                />
              </div>
              <div className="flex-1 space-y-1 w-full">
                <button
                  onClick={fetchDynamicMetrics}
                  disabled={isComputing || !factorDatasetReady}
                  className="w-full h-[30px] bg-blue-700 hover:bg-blue-600 text-white text-xs font-bold uppercase tracking-wider disabled:opacity-50"
                >
                  {isComputing ? 'COMPUTING...' : 'RUN BACKTEST'}
                </button>
              </div>
            </div>

            {/* Row 2: Universe Selection */}
            <div className="border-t border-gray-800 pt-3">
              <div className="flex justify-between items-center mb-2">
                <label className="text-[10px] font-bold uppercase text-gray-500">Universe Selection</label>
                <div className="flex gap-2">
                  <button onClick={() => setBacktestMode('all')} className={`text-[10px] px-2 py-1 uppercase font-bold border ${backtestMode === 'all' ? 'bg-amber-600 border-amber-600 text-white' : 'border-gray-600 text-gray-500'}`}>All Factors</button>
                  <button onClick={() => setBacktestMode('selected')} className={`text-[10px] px-2 py-1 uppercase font-bold border ${backtestMode === 'selected' ? 'bg-amber-600 border-amber-600 text-white' : 'border-gray-600 text-gray-500'}`}>Custom Selection</button>
                </div>
              </div>
              {backtestMode === 'selected' && (
                <div className="bg-gray-100 dark:bg-[#151515] p-2 border border-gray-300 dark:border-gray-800">
                  <div className="flex justify-between text-[10px] text-gray-500 mb-2 uppercase">
                    <span>{selectedFactors.length} Selected (Equal Weight)</span>
                    <div className="space-x-2">
                      <button onClick={() => setSelectedFactors(Object.keys(factorPnL[0] || {}).filter(k => k.startsWith('factor_')))} className="hover:text-white">Select All</button>
                      <button onClick={() => setSelectedFactors([])} className="hover:text-white">Clear</button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1 max-h-[100px] overflow-y-auto">
                    {Object.keys(factorPnL[0] || {}).filter(k => k.startsWith('factor_')).map(f => (
                      <button
                        key={f}
                        onClick={() => toggleFactor(f)}
                        className={`text-[10px] px-2 py-0.5 border ${selectedFactors.includes(f) ? 'bg-amber-900/30 border-amber-600 text-amber-500' : 'bg-transparent border-gray-700 text-gray-500'}`}
                      >
                        {f.replace('factor_', 'F')}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          {computeError && <div className="mt-2 text-[10px] text-red-500 font-mono bg-red-900/20 p-1 border border-red-900">{computeError}</div>}
        </BBCard>

        {/* --- Metrics Ticker --- */}
        {metrics.length > 0 && (
          <div className="col-span-12 border-y border-gray-800 bg-[#111] py-2 overflow-x-auto">
            <div className="flex gap-8 px-4 min-w-max">
              <MetricValue label="NAV (Return)" value={((metrics.find(m => m.Dataset === "Test")?.ret ?? 0) * 100).toFixed(2) + "%"} color="text-amber-500" />
              <MetricValue label="Sharpe (Test)" value={(metrics.find(m => m.Dataset === "Test")?.ret_sharpe as number)?.toFixed(2) ?? "N/A"} color="text-gray-100" />
              <MetricValue label="IC (Test)" value={(metrics.find(m => m.Dataset === "Test")?.ic as number)?.toFixed(4) ?? "N/A"} color="text-gray-100" />
              <MetricValue label="Max Drawdown" value={((metrics.find(m => m.Dataset === "Test")?.ret_mdd ?? 0) * 100).toFixed(2) + "%"} color="text-red-500" />
              <MetricValue label="Val. Sharpe" value={(metrics.find(m => m.Dataset === "Validation")?.ret_sharpe as number)?.toFixed(2) ?? "N/A"} sub="In-Sample" color="text-gray-100" />
            </div>
          </div>
        )}

        {/* --- Main Chart (Holding Period) --- */}
        <BBCard title="EQUITY CURVE & PERFORMANCE" icon={TrendingUp} span="col-span-12 lg:col-span-8" className="min-h-[400px]">
          {holdingsPnL.length > 0 ? (
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={holdingsPnL}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#666' }} interval="preserveStartEnd" minTickGap={30} />
                  <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={(val) => val.toFixed(0) + '%'} width={40} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#000', borderColor: '#333', color: '#fff', fontSize: '12px' }}
                    itemStyle={{ color: '#fff' }}
                    labelStyle={{ color: '#aaa', marginBottom: '5px' }}
                    formatter={(value: number) => [`${value.toFixed(2)}%`]}
                  />
                  <Legend verticalAlign="top" height={36} iconType="rect" iconSize={10} wrapperStyle={{ fontSize: '10px', textTransform: 'uppercase' }} />
                  <Line type="stepAfter" dataKey="cumulative_ret" name="Strategy Equity" stroke="#FF5500" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#FF5500' }} />
                  <Line type="monotone" dataKey="cumulative_benchmark" name="Benchmark" stroke="#666" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-gray-500 text-xs font-mono uppercase">Waiting for Backtest Run...</div>
          )}
        </BBCard>

        {/* --- Side Panel: Detailed Stats --- */}
        <BBCard title="PERFORMANCE STATISTICS" icon={BarChart3} span="col-span-12 lg:col-span-4">
          <table className="w-full text-xs font-mono">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr><th className="text-left py-2 font-normal">METRIC</th><th className="text-right py-2 font-normal">TEST</th><th className="text-right py-2 font-normal">VALID</th></tr>
            </thead>
            <tbody className="divide-y divide-gray-900">
              {metrics.length > 0 && ["ic", "icir", "ret_sharpe", "ret_mdd", "ric"].map(k => (
                <tr key={k}>
                  <td className="py-2 text-gray-400 uppercase">{k.replace('ret_', '')}</td>
                  <td className="py-2 text-right text-white">{(metrics.find(m => m.Dataset === "Test")?.[k as keyof StrategyMetric] as number)?.toFixed(3)}</td>
                  <td className="py-2 text-right text-gray-500">{(metrics.find(m => m.Dataset === "Validation")?.[k as keyof StrategyMetric] as number)?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 pt-4 border-t border-gray-800">
            <h4 className="text-[10px] font-bold uppercase text-gray-500 mb-2">Filters</h4>
            <div className="flex items-center gap-2">
              <input type="date" className="flex-1 bg-gray-900 border border-gray-700 text-white text-xs p-1 font-mono" value={dateRange.startDate} onChange={e => setDateRange({ ...dateRange, startDate: e.target.value })} />
              <span className="text-gray-500 text-xs">-</span>
              <input type="date" className="flex-1 bg-gray-900 border border-gray-700 text-white text-xs p-1 font-mono" value={dateRange.endDate} onChange={e => setDateRange({ ...dateRange, endDate: e.target.value })} />
            </div>
          </div>
        </BBCard>

        {/* --- Correlation Matrix (Restored) --- */}
        {(correlationMatrix || vifScores.length > 0) && (
          <BBCard title="CORRELATION & RISKS" icon={Grid3X3} span="col-span-12 lg:col-span-6">
            <div className="flex flex-col gap-4">
              {correlationMatrix && correlationMatrix.matrix.length > 0 && (
                <div>
                  <h4 className="text-[10px] uppercase font-bold text-gray-500 mb-2">Correlation Matrix (Top 10)</h4>
                  <div className="overflow-x-auto">
                    <table className="text-[10px] font-mono w-full table-fixed">
                      <thead>
                        <tr><th className="w-6"></th>{correlationMatrix.factors.slice(0, 10).map((f, i) => <th key={i} className="text-gray-500">{i}</th>)}</tr>
                      </thead>
                      <tbody>
                        {correlationMatrix.matrix.slice(0, 10).map((row, i) => (
                          <tr key={i}>
                            <td className="text-gray-500 font-bold">{i}</td>
                            {row.slice(0, 10).map((val, j) => (
                              <td key={j} className="p-0.5 text-center" style={{ backgroundColor: val > 0.7 ? '#330000' : val > 0.5 ? '#221100' : 'transparent', color: val > 0.5 ? '#ffaa00' : '#555' }}>
                                {val.toFixed(1).replace('0.', '.')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </BBCard>
        )}

        {/* --- VIF Analysis (Restored) --- */}
        {vifScores.length > 0 && (
          <BBCard title="MULTICOLLINEARITY (VIF)" icon={AlertTriangle} span="col-span-12 lg:col-span-6">
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={vifScores.slice(0, 15)} layout="vertical" margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#333" />
                  <XAxis type="number" stroke="#555" tick={{ fontSize: 9 }} />
                  <YAxis type="category" dataKey="factor_id" stroke="#555" tick={{ fontSize: 9 }} width={30} />
                  <Tooltip cursor={{ fill: '#222' }} contentStyle={{ backgroundColor: '#000', borderColor: '#333', fontSize: '10px' }} />
                  <Bar dataKey="vif" name="VIF Score" barSize={10}>
                    {vifScores.slice(0, 15).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.multicollinear ? '#ef4444' : '#22c55e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-gray-500 mt-2">Factors with VIF &gt; 10 are significantly correlated with others.</p>
          </BBCard>
        )}

        {/* --- Individual Factor Analysis (Restored) --- */}
        <BBCard title="FACTOR DEEP DIVE & COMPARISON" icon={Search} span="col-span-12">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-gray-800 pb-2 gap-2">
              <div className="flex gap-2 shrink-0">
                <button onClick={() => { setShowCumulative(true); setFactorViewMode('returns'); }} className={`text-[10px] px-2 py-1 uppercase font-bold ${showCumulative ? 'bg-amber-600 text-white' : 'text-gray-500 bg-gray-900 border border-gray-700'}`}>Cumulative</button>
                <button onClick={() => { setShowCumulative(false); setFactorViewMode('returns'); }} className={`text-[10px] px-2 py-1 uppercase font-bold ${!showCumulative ? 'bg-amber-600 text-white' : 'text-gray-500 bg-gray-900 border border-gray-700'}`}>Daily</button>
              </div>
              <div className="flex flex-wrap gap-1 items-center justify-end w-full">
                {selectedFactors.length > 0 ? (
                  <>
                    <span className="text-[10px] text-gray-500 uppercase mr-2 hidden sm:inline">Comparing:</span>
                    {selectedFactors.map(f => (
                      <button key={f} onClick={() => toggleFactor(f)} className="text-[10px] px-2 py-1 uppercase whitespace-nowrap border bg-amber-900/40 border-amber-600 text-amber-500 flex items-center gap-1 hover:bg-amber-900/60">
                        {f.replace('factor_', 'F')} <span className="text-[8px]">✕</span>
                      </button>
                    ))}
                    <button onClick={() => setSelectedFactors([])} className="text-[10px] px-2 py-1 uppercase border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 ml-1">
                      Clear All
                    </button>
                  </>
                ) : (
                  <span className="text-[10px] text-gray-500 italic flex items-center gap-1">
                    <Info className="h-3 w-3" /> Select factors from the Universe table below to compare
                  </span>
                )}
              </div>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={showCumulative ? cumulativeFactorPnL : factorPnL}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#222" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#666' }} minTickGap={30} />
                  <YAxis tick={{ fontSize: 10, fill: '#666' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#000', borderColor: '#333', fontSize: '10px' }} />
                  {selectedFactors.length === 0 ? (
                    /* Default show first 5 if none selected, as background context */
                    Object.keys(factorPnL[0] || {}).filter(k => k.startsWith('factor_')).slice(0, 5).map((k, i) => (
                      <Line key={k} type="monotone" dataKey={k} stroke={`hsl(0, 0%, ${30 + i * 10}%)`} dot={false} strokeWidth={1} strokeDasharray="3 3" name={`${k.replace('factor_', 'F')} (Sample)`} />
                    ))
                  ) : (
                    selectedFactors.map((k, i) => (
                      <Line key={k} type="monotone" dataKey={k} stroke={`hsl(${50 + (i * 137.5) % 360}, 70%, 50%)`} dot={false} strokeWidth={2} name={k.replace('factor_', 'F')} />
                    ))
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </BBCard>

        {/* --- Advanced Analysis Section Divider --- */}
        <div className="col-span-12 my-2 border-b-2 border-dashed border-gray-800" />

        {/* --- Factor Analysis / Dictionary --- */}
        <BBCard title="FACTOR UNIVERSE SELECTION" icon={BookOpen} span="col-span-12 lg:col-span-8">
          <div className="overflow-x-auto max-h-[600px]">
            <table className="w-full text-xs font-mono">
              <thead className="bg-gray-100 dark:bg-[#1a1a1a] sticky top-0 z-10">
                <tr>
                  <th className="px-2 py-2 text-center text-gray-500 font-normal w-8">
                    <input
                      type="checkbox"
                      className="accent-amber-600"
                      checked={factorDictionary.length > 0 && selectedFactors.length === factorDictionary.length}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedFactors(factorDictionary.map(f => `factor_${f.factor_id}`));
                          setBacktestMode('selected');
                        } else {
                          setSelectedFactors([]);
                        }
                      }}
                    />
                  </th>
                  <th className="px-2 py-2 text-left text-gray-500 font-normal cursor-pointer hover:text-white" onClick={() => setSortConfig({ key: 'factor_id', direction: sortConfig.key === 'factor_id' && sortConfig.direction === 'asc' ? 'desc' : 'asc' })}>
                    ID {sortConfig.key === 'factor_id' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-2 py-2 text-left text-gray-500 font-normal">NAME/EXPR</th>
                  <th className="px-2 py-2 text-left text-gray-500 font-normal">TYPE</th>
                  <th className="px-2 py-2 text-right text-gray-500 font-normal cursor-pointer hover:text-white" onClick={() => setSortConfig({ key: 'ic', direction: sortConfig.key === 'ic' && sortConfig.direction === 'desc' ? 'asc' : 'desc' })}>
                    IC {sortConfig.key === 'ic' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-2 py-2 text-right text-gray-500 font-normal cursor-pointer hover:text-white" onClick={() => setSortConfig({ key: 'sharpe', direction: sortConfig.key === 'sharpe' && sortConfig.direction === 'desc' ? 'asc' : 'desc' })}>
                    SHARPE {sortConfig.key === 'sharpe' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-2 py-2 text-right text-gray-500 font-normal">DETAILS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {[...factorDictionary].sort((a, b) => {
                  const key = sortConfig.key as keyof FactorInfo;
                  const valA = (a[key] ?? -999) as number;
                  const valB = (b[key] ?? -999) as number;
                  if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
                  if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
                  return 0;
                }).map(f => {
                  const fKey = `factor_${f.factor_id}`;
                  const isSelected = selectedFactors.includes(fKey);
                  return (
                    <tr key={f.factor_id} className={`hover:bg-gray-900 transition-colors ${isSelected ? 'bg-amber-950/20' : ''}`}>
                      <td className="px-2 py-1.5 text-center">
                        <input
                          type="checkbox"
                          className="accent-amber-600 cursor-pointer"
                          checked={isSelected}
                          onChange={() => {
                            toggleFactor(fKey);
                            // Auto-enable 'Custom Selection' mode when user manually selects a factor
                            if (!isSelected && backtestMode !== 'selected') {
                              setBacktestMode('selected');
                            }
                          }}
                        />
                      </td>
                      <td className="px-2 py-1.5 text-amber-600 font-bold">{f.factor_id}</td>
                      <td className="px-2 py-1.5 truncate max-w-[200px]" title={f.name}>
                        {f.expr && f.expr !== 'Expression unavailable' ? (
                          <span className="font-mono text-gray-400">{f.expr}</span>
                        ) : (
                          <span className="text-gray-500">{f.name}</span>
                        )}
                      </td>
                      <td className="px-2 py-1.5 text-gray-500">{f.type}</td>
                      <td className="px-2 py-1.5 text-right font-bold text-white">{f.ic?.toFixed(3)}</td>
                      <td className="px-2 py-1.5 text-right text-gray-400">{f.sharpe?.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right">
                        <button
                          onClick={(e) => { e.stopPropagation(); setSelectedFactorDetail(f); setShowDetailPanel(true); }}
                          className="text-[10px] px-2 py-0.5 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 uppercase rounded-sm"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </BBCard>

        {/* --- Exports & Utils --- */}
        <BBCard title="DATA UTILS" icon={Download} span="col-span-12 lg:col-span-4">
          <div className="space-y-3">
            <button onClick={exportRawFactorValues} className="w-full py-2 border border-gray-700 hover:bg-gray-900 text-gray-300 text-xs font-mono uppercase text-left px-3 flex justify-between">
              <span>Per-Stock Factor Values (.csv)</span> <Download className="h-3 w-3" />
            </button>
            <button onClick={exportFactorDictionary} className="w-full py-2 border border-gray-700 hover:bg-gray-900 text-gray-300 text-xs font-mono uppercase text-left px-3 flex justify-between">
              <span>Factor Dictionary (.json)</span> <Download className="h-3 w-3" />
            </button>
          </div>

          {/* Detail Panel Float */}
          {showDetailPanel && selectedFactorDetail && (
            <div className="mt-4 p-3 border border-amber-900 bg-amber-950/20 text-xs text-amber-500 font-mono">
              <div className="flex justify-between font-bold mb-2">
                <span>SELECTED: F{selectedFactorDetail.factor_id}</span>
                <button onClick={() => setShowDetailPanel(false)} className="hover:text-white">X</button>
              </div>
              <p className="break-all">{selectedFactorDetail.expr}</p>
            </div>
          )}
        </BBCard>

      </div>
    </div>
  );
}
