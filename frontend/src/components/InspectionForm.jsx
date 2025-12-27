import React from 'react';
import { Settings, Gauge, Fuel, Calendar, Zap, Droplets, Volume2, Wind, Disc, Activity } from 'lucide-react';
import { DiagnosticCheckbox } from './DiagnosticCheckbox';

export function InspectionForm({ formData, handleInputChange, handleCheckboxChange, handleSubmit, loading }) {
    return (
        <section className="lg:col-span-7 space-y-6 animate-fade-in-up">
            <div className="bg-white dark:bg-gray-800/50 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-xl">
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                    <Settings className="w-5 h-5 text-blue-500" />
                    Inspection Parameters
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Odometer */}
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-600 dark:text-gray-300">Odometer Reading (km)</label>
                            <div className="relative group">
                                <Gauge className="absolute left-3 top-3.5 w-5 h-5 text-gray-400 group-focus-within:text-blue-500 transition-colors" />
                                <input
                                    type="number"
                                    name="odometer_reading"
                                    value={formData.odometer_reading}
                                    onChange={handleInputChange}
                                    className="w-full pl-10 pr-4 py-3 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                    placeholder="e.g. 50000"
                                />
                            </div>
                        </div>

                        {/* Fuel Type */}
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-600 dark:text-gray-300">Fuel Type</label>
                            <div className="relative group">
                                <Fuel className="absolute left-3 top-3.5 w-5 h-5 text-gray-400 group-focus-within:text-blue-500 transition-colors" />
                                <select
                                    name="fuel_type"
                                    value={formData.fuel_type}
                                    onChange={handleInputChange}
                                    className="w-full pl-10 pr-4 py-3 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none appearance-none transition-all cursor-pointer"
                                >
                                    <option>Petrol</option>
                                    <option>Diesel</option>
                                    <option>CNG</option>
                                    <option>LPG</option>
                                    <option>Electric</option>
                                    <option>Hybrid</option>
                                </select>
                            </div>
                        </div>

                        {/* Date */}
                        <div className="space-y-2 md:col-span-2">
                            <label className="text-sm font-medium text-gray-600 dark:text-gray-300">Inspection Time</label>
                            <div className="relative group">
                                <Calendar className="absolute left-3 top-3.5 w-5 h-5 text-gray-400 group-focus-within:text-blue-500 transition-colors" />
                                <input
                                    type="datetime-local"
                                    name="inspectionStartTime"
                                    value={formData.inspectionStartTime}
                                    onChange={handleInputChange}
                                    className="w-full pl-10 pr-4 py-3 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-4 pt-4">
                        <label className="text-sm font-medium text-gray-600 dark:text-gray-300 block mb-2">Diagnostic Checks</label>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <DiagnosticCheckbox
                                label="Battery Jump Start"
                                icon={Zap}
                                checked={formData.diagnostics.battery_jump_start}
                                onChange={() => handleCheckboxChange('battery_jump_start')}
                            />
                            <DiagnosticCheckbox
                                label="Engine Oil Leak"
                                icon={Droplets}
                                checked={formData.diagnostics.engine_oil_leak}
                                onChange={() => handleCheckboxChange('engine_oil_leak')}
                            />
                            <DiagnosticCheckbox
                                label="Abnormal Sound"
                                icon={Volume2}
                                checked={formData.diagnostics.engine_sound_abnormal}
                                onChange={() => handleCheckboxChange('engine_sound_abnormal')}
                            />
                            <DiagnosticCheckbox
                                label="White Smoke"
                                icon={Wind}
                                checked={formData.diagnostics.exhaust_smoke_white}
                                onChange={() => handleCheckboxChange('exhaust_smoke_white')}
                            />
                            <DiagnosticCheckbox
                                label="Hard Clutch"
                                icon={Disc}
                                checked={formData.diagnostics.clutch_hard}
                                onChange={() => handleCheckboxChange('clutch_hard')}
                            />
                            <DiagnosticCheckbox
                                label="Gear Shift Issue"
                                icon={Settings}
                                checked={formData.diagnostics.gear_shifting_hard}
                                onChange={() => handleCheckboxChange('gear_shifting_hard')}
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-4 mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-blue-500/30 transform transition-all active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                Analyzing Engine...
                            </>
                        ) : (
                            <>
                                Run AI Prediction
                                <Activity className="w-5 h-5" />
                            </>
                        )}
                    </button>
                </form>
            </div>
        </section>
    );
}
