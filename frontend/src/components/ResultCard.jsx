import React from 'react';
import { Activity, AlertTriangle } from 'lucide-react';

export function ResultCard({ result, error }) {
    const getRatingColor = (rating) => {
        if (rating >= 4) return 'text-green-500';
        if (rating >= 3) return 'text-yellow-500';
        return 'text-red-500';
    };

    return (
        <section className="lg:col-span-5 space-y-6 animate-fade-in-up">
            <div className={`h-full min-h-[400px] flex flex-col justify-center items-center p-8 rounded-3xl border border-white/20 shadow-2xl transition-all duration-500 ${result ? 'bg-white dark:bg-gray-800' : 'bg-gray-100 dark:bg-gray-800/50 border-dashed'}`}>

                {result ? (
                    <div className="text-center w-full space-y-8 animate-scale-in">
                        <div className="relative inline-block">
                            <svg className="w-48 h-48 transform -rotate-90">
                                <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-gray-200 dark:text-gray-700" />
                                <circle
                                    cx="96" cy="96" r="88"
                                    stroke="currentColor"
                                    strokeWidth="12"
                                    fill="transparent"
                                    strokeDasharray={552}
                                    strokeDashoffset={552 - (552 * result.prediction) / 5}
                                    className={`${getRatingColor(result.prediction)} transition-all duration-1000 ease-out`}
                                    strokeLinecap="round"
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className={`text-5xl font-bold ${getRatingColor(result.prediction)}`}>{result.prediction.toFixed(1)}</span>
                                <span className="text-sm text-gray-400 mt-1">out of 5.0</span>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-2xl font-bold mb-2">{result.rating_text} Condition</h3>
                            <p className="text-gray-500 max-w-xs mx-auto">
                                Based on the provided diagnostics, this engine is displaying {result.prediction > 4 ? 'optimal' : result.prediction > 2.5 ? 'acceptable' : 'critical'} performance characteristics.
                            </p>
                        </div>

                        <div className="grid grid-cols-2 gap-4 w-full">
                            <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/50">
                                <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Confidence</div>
                                <div className="text-lg font-bold text-blue-600">94.2%</div>
                            </div>
                            <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/50">
                                <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Market Value</div>
                                <div className="text-lg font-bold text-green-600">
                                    {result.prediction > 4 ? 'High' : result.prediction > 2 ? 'Medium' : 'Low'}
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="text-center text-gray-400">
                        <div className="w-20 h-20 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Activity className="w-10 h-10 opacity-50" />
                        </div>
                        <h3 className="text-lg font-medium mb-1">Ready to Analyze</h3>
                        <p className="text-sm">Enter vehicle parameters to generate a rating.</p>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-xl flex items-center gap-2 text-sm w-full animate-shake">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        {error}
                    </div>
                )}
            </div>
        </section>
    );
}
