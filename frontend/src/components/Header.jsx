import React from 'react';
import { Car, Gauge } from 'lucide-react';

export function Header() {
    return (
        <header className="max-w-6xl mx-auto mb-12 flex items-center justify-between animate-fade-in-down">
            <div className="flex items-center gap-3">
                <div className="p-3 bg-blue-600 rounded-2xl shadow-lg shadow-blue-500/30">
                    <Car className="w-8 h-8 text-white" />
                </div>
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">EngineRater<span className="text-blue-600">.AI</span></h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Advanced Vehicle Diagnostics System</p>
                </div>
            </div>
            <button className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full border border-gray-200 dark:border-gray-700 hover:bg-white dark:hover:bg-gray-800 transition-all font-medium text-sm">
                <Gauge className="w-4 h-4" />
                Analytics Dashboard
            </button>
        </header>
    );
}
