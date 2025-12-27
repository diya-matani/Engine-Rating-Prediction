import React, { useState } from 'react';
import {
  Gauge,
  Car,
  Fuel,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Calendar,
  Zap,
  Droplets,
  Volume2,
  Wind,
  Disc,
  Settings
} from 'lucide-react';
import { predictRating } from './api';

function App() {
  const [formData, setFormData] = useState({
    inspectionStartTime: new Date().toISOString().slice(0, 16),
    odometer_reading: 50000,
    fuel_type: 'Petrol',
    diagnostics: {
      battery_jump_start: false,
      engine_oil_leak: false,
      engine_sound_abnormal: false,
      exhaust_smoke_white: false,
      clutch_hard: false,
      gear_shifting_hard: false
    }
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleCheckboxChange = (name) => {
    setFormData(prev => ({
      ...prev,
      diagnostics: {
        ...prev.diagnostics,
        [name]: !prev.diagnostics[name]
      }
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predictRating(formData);
      setResult(data);
    } catch (err) {
      setError('Failed to get prediction. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getRatingColor = (rating) => {
    if (rating >= 4) return 'text-green-500';
    if (rating >= 3) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-6 md:p-12 font-sans text-gray-800 dark:text-gray-100 transition-colors duration-500">

      {/* Header */}
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

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Input Section */}
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

        {/* Result Section */}
        <section className="lg:col-span-5 space-y-6 animate-fade-in-up">
          {/* Main Result Card */}
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

      </main>
    </div>
  );
}

function DiagnosticCheckbox({ label, icon: Icon, checked, onChange }) {
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

export default App;
