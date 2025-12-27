import React from 'react';
import { CheckCircle2 } from 'lucide-react';

export function DiagnosticCheckbox({ label, icon: Icon, checked, onChange }) {
    return (
        <div
            onClick={onChange}
            className={`relative flex items-center gap-3 p-4 rounded-xl border transition-all cursor-pointer select-none group
      ${checked
                    ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 ring-1 ring-blue-500/50'
                    : 'bg-white/50 dark:bg-gray-800/30 border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
                }`}
        >
            <div className={`p-2 rounded-lg transition-colors ${checked ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-400 group-hover:bg-blue-100 group-hover:text-blue-500'}`}>
                <Icon className="w-4 h-4" />
            </div>
            <span className={`font-medium text-sm transition-colors ${checked ? 'text-blue-800 dark:text-blue-200' : 'text-gray-600 dark:text-gray-400'}`}>
                {label}
            </span>
            {checked && <CheckCircle2 className="w-4 h-4 text-blue-500 ml-auto" />}
        </div>
    );
}
