import React, { useState } from 'react';
import { predictRating } from './api';
import { Header } from './components/Header';
import { InspectionForm } from './components/InspectionForm';
import { ResultCard } from './components/ResultCard';

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-6 md:p-12 font-sans text-gray-800 dark:text-gray-100 transition-colors duration-500">
      <Header />
      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        <InspectionForm
          formData={formData}
          handleInputChange={handleInputChange}
          handleCheckboxChange={handleCheckboxChange}
          handleSubmit={handleSubmit}
          loading={loading}
        />
        <ResultCard result={result} error={error} />
      </main>
    </div>
  );
}

export default App;
