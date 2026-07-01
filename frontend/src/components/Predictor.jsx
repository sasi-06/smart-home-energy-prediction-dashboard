import React, { useState } from 'react';
import axios from 'axios';

const APPLIANCES = ["Fridge", "Oven", "Dishwasher", "Heater", "Microwave", "Air Conditioning", "Computer", "TV", "Lights", "Washing Machine"];
const SEASONS = ["Spring", "Summer", "Fall", "Winter"];

export default function Predictor() {
  const [formData, setFormData] = useState({
    temperature: 15.0,
    household_size: 3,
    hour: 14,
    month: 7,
    day_of_week: 2,
    appliance_types: ["Oven", "TV"],
    season: "Summer"
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleApplianceChange = (app) => {
    setFormData(prev => {
      const current = prev.appliance_types;
      if (current.includes(app)) {
        return { ...prev, appliance_types: current.filter(a => a !== app) };
      } else {
        return { ...prev, appliance_types: [...current, app] };
      }
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.appliance_types.length === 0) {
      setError("Please select at least one appliance.");
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.post(`${apiUrl}/api/predict`, formData);
      setResult(res.data);
      // Save to localStorage for Bill Calculator
      localStorage.setItem('ml_predictions', JSON.stringify(res.data.predictions));
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const calculateTNBill = (kwh) => {
    // simplified TN tariff for demo (bi-monthly)
    const units = kwh * 30; // approx per month for display
    let cost = 0;
    if (units <= 500) {
      if (units > 100) cost += Math.min(units - 100, 100) * 2.25;
      if (units > 200) cost += Math.min(units - 200, 200) * 4.50;
      if (units > 400) cost += (units - 400) * 6.00;
    } else {
      const base_cost_500 = (300 * 4.50) + (100 * 6.00);
      cost += base_cost_500;
      if (units > 500) cost += Math.min(units - 500, 100) * 8.00;
      if (units > 600) cost += Math.min(units - 600, 200) * 9.00;
      if (units > 800) cost += Math.min(units - 800, 200) * 10.00;
      if (units > 1000) cost += (units - 1000) * 11.00;
    }
    return cost.toFixed(2);
  };

  return (
    <div className="glass-panel">
      <h2 style={{ marginBottom: '20px', color: 'var(--accent-primary)' }}>Predict Future Energy Usage</h2>
      
      <form onSubmit={handleSubmit} className="form-grid">
        <div>
          <div className="form-group">
            <label>Outdoor Temperature (°C): {formData.temperature}</label>
            <input 
              type="range" 
              min="-20" max="50" step="0.5" 
              value={formData.temperature} 
              onChange={e => setFormData({...formData, temperature: parseFloat(e.target.value)})}
              style={{ width: '100%' }}
            />
          </div>
          <div className="form-group">
            <label>Household Size</label>
            <input 
              type="number" 
              className="form-control"
              min="1" max="10" 
              value={formData.household_size} 
              onChange={e => setFormData({...formData, household_size: parseInt(e.target.value)})}
            />
          </div>
          <div className="form-group">
            <label>Appliances</label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
              {APPLIANCES.map(app => (
                <div 
                  key={app}
                  onClick={() => handleApplianceChange(app)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '20px',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                    background: formData.appliance_types.includes(app) ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)',
                    border: `1px solid ${formData.appliance_types.includes(app) ? 'var(--accent-primary)' : 'var(--border-color)'}`
                  }}
                >
                  {app}
                </div>
              ))}
            </div>
          </div>
        </div>
        
        <div>
          <div className="form-group">
            <label>Season</label>
            <select 
              className="form-control"
              value={formData.season}
              onChange={e => setFormData({...formData, season: e.target.value})}
            >
              {SEASONS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Time of Day (Hour 0-23)</label>
            <input 
              type="number" 
              className="form-control"
              min="0" max="23" 
              value={formData.hour} 
              onChange={e => setFormData({...formData, hour: parseInt(e.target.value)})}
            />
          </div>
          <div className="form-group">
            <label>Month (1-12)</label>
            <input 
              type="number" 
              className="form-control"
              min="1" max="12" 
              value={formData.month} 
              onChange={e => setFormData({...formData, month: parseInt(e.target.value)})}
            />
          </div>
          <div className="form-group">
            <label>Day of Week (0=Mon, 6=Sun)</label>
            <input 
              type="number" 
              className="form-control"
              min="0" max="6" 
              value={formData.day_of_week} 
              onChange={e => setFormData({...formData, day_of_week: parseInt(e.target.value)})}
            />
          </div>
        </div>

        <div style={{ gridColumn: '1 / -1', marginTop: '10px' }}>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Analyzing Parameters...' : 'Predict Energy Consumption'}
          </button>
        </div>
      </form>

      {error && <div className="error-message" style={{marginTop: '20px'}}>{error}</div>}

      {result && (
        <div style={{ marginTop: '30px', animation: 'fadeIn 0.5s ease' }}>
          <div className="metrics-container">
            <div className="metric-card success glass-panel">
              <div className="metric-label">Total Predicted Consumption</div>
              <div className="metric-value">
                {result.total_kwh_per_day.toFixed(2)} <span className="metric-unit">kWh/day</span>
              </div>
            </div>
            <div className="metric-card warning glass-panel">
              <div className="metric-label">Estimated Cost / Month (TN Tariff)</div>
              <div className="metric-value">
                ₹{calculateTNBill(result.total_kwh_per_day)}
              </div>
            </div>
          </div>
          
          <h3 style={{ marginTop: '20px', marginBottom: '15px' }}>Breakdown by Appliance</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px' }}>
            {Object.entries(result.predictions).map(([app, val]) => (
              <div key={app} style={{ 
                background: 'rgba(30,41,59,0.5)', 
                padding: '15px', 
                borderRadius: '10px',
                border: '1px solid var(--border-color)',
                minWidth: '150px'
              }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{app}</div>
                <div style={{ fontSize: '1.5rem', color: 'var(--accent-primary)', fontWeight: 'bold' }}>
                  {val.toFixed(2)} <span style={{ fontSize: '0.8rem' }}>kWh</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
