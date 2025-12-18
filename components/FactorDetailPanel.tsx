"use client";

import React from "react";
import { FactorInfo } from "@/utils/data";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { X, Code, Tag, Info, TrendingUp, TrendingDown, Zap } from "lucide-react";

interface FactorDetailPanelProps {
    factor: FactorInfo | null;
    onClose: () => void;
}

export function FactorDetailPanel({ factor, onClose }: FactorDetailPanelProps) {
    if (!factor) return null;

    const typeColors: Record<string, string> = {
        momentum: "bg-purple-100 text-purple-800",
        volume: "bg-blue-100 text-blue-800",
        volatility: "bg-orange-100 text-orange-800",
        mean_reversion: "bg-green-100 text-green-800",
        correlation: "bg-cyan-100 text-cyan-800",
        technical: "bg-yellow-100 text-yellow-800",
        composite: "bg-gray-100 text-gray-800",
    };

    const typeColor = typeColors[factor.type || 'composite'] || typeColors.composite;

    return (
        <Card className="h-full border-l-4 border-l-blue-500 shadow-lg animate-in slide-in-from-right duration-300">
            <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                    <div className="space-y-1">
                        <CardTitle className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                            {factor.name}
                        </CardTitle>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${typeColor}`}>
                            <Tag className="h-3 w-3 mr-1" />
                            {factor.type || 'composite'}
                        </span>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition-colors"
                        aria-label="Close detail panel"
                    >
                        <X className="h-5 w-5 text-gray-500" />
                    </button>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Expression */}
                <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center gap-1.5 mb-2">
                        <Code className="h-4 w-4" />
                        Expression
                    </h4>
                    <code className="block bg-gray-100 dark:bg-gray-800 p-3 rounded-lg text-xs font-mono break-all text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700">
                        {factor.expr}
                    </code>
                </div>

                {/* Description */}
                <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center gap-1.5 mb-2">
                        <Info className="h-4 w-4" />
                        Description
                    </h4>
                    <p className="text-sm text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 p-3 rounded-lg">
                        {factor.description}
                    </p>
                </div>

                {/* IC Metrics */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="flex items-center gap-1.5 mb-1">
                            <TrendingUp className="h-4 w-4 text-blue-600" />
                            <span className="text-xs text-gray-600 dark:text-gray-400">IC</span>
                        </div>
                        <p className="text-xl font-bold text-blue-700 dark:text-blue-400">
                            {factor.ic?.toFixed(4) ?? 'N/A'}
                        </p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-3 rounded-lg border border-purple-200 dark:border-purple-800">
                        <div className="flex items-center gap-1.5 mb-1">
                            <Zap className="h-4 w-4 text-purple-600" />
                            <span className="text-xs text-gray-600 dark:text-gray-400">ICIR</span>
                        </div>
                        <p className="text-xl font-bold text-purple-700 dark:text-purple-400">
                            {factor.icir?.toFixed(4) ?? 'N/A'}
                        </p>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
                        <div className="flex items-center gap-1.5 mb-1">
                            <TrendingUp className="h-4 w-4 text-green-600" />
                            <span className="text-xs text-gray-600 dark:text-gray-400">Sharpe</span>
                        </div>
                        <p className="text-xl font-bold text-green-700 dark:text-green-400">
                            {factor.sharpe?.toFixed(2) ?? 'N/A'}
                        </p>
                    </div>
                    <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
                        <div className="flex items-center gap-1.5 mb-1">
                            <TrendingDown className="h-4 w-4 text-orange-600" />
                            <span className="text-xs text-gray-600 dark:text-gray-400">MDD</span>
                        </div>
                        <p className="text-xl font-bold text-orange-700 dark:text-orange-400">
                            {factor.mdd ? (factor.mdd * 100).toFixed(1) + '%' : 'N/A'}
                        </p>
                    </div>
                </div>

                {/* Source */}
                <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>Factor ID: {factor.factor_id}</span>
                        <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded">
                            Source: {factor.source ?? 'unknown'}
                        </span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
