import React, { useState, useEffect } from 'react';

const DEFAULT_ROWS = [
  { id: 1, appliance: "Air Conditioner", qty: 1, watts: 1500, hours: 8 },
  { id: 2, appliance: "Refrigerator", qty: 1, watts: 200, hours: 24 },
  { id: 3, appliance: "Television", qty: 1, watts: 100, hours: 4 },
  { id: 4, appliance: "Ceiling Fan", qty: 4, watts: 75, hours: 12 },
  { id: 5, appliance: "LED Light", qty: 10, watts: 12, hours: 6 },
];

export default function BillCalculator() {
  const [rows, setRows] = useState(DEFAULT_ROWS);
  const [billingDays, setBillingDays] = useState(30);

  useEffect(() => {
    // Check if we want to import ML predictions
    const checkImport = () => {
      const mlData = localStorage.getItem('ml_predictions');
      if (mlData) {
        try {
          const parsed = JSON.parse(mlData);
          const newRows = Object.entries(parsed).map(([app, kwh], idx) => ({
            id: Date.now() + idx,
            appliance: `${app} (Predicted)`,
            qty: 1,
            watts: 1000, // We use 1000W so 1hr = 1kWh directly based on the prediction
            hours: kwh
          }));
          if (newRows.length > 0) {
            setRows(newRows);
          }
        } catch (e) {
          console.error("Failed to parse ML predictions", e);
        }
      }
    };
    
    // Only auto-import on mount if explicitly clicked or just provide a button
    // We'll just provide a button for the user to pull it
  }, []);

  const importMLPredictions = () => {
    const mlData = localStorage.getItem('ml_predictions');
    if (mlData) {
      try {
        const parsed = JSON.parse(mlData);
        const newRows = Object.entries(parsed).map(([app, kwh], idx) => ({
          id: Date.now() + idx,
          appliance: `${app} (Predicted)`,
          qty: 1,
          watts: 1000,
          hours: kwh
        }));
        if (newRows.length > 0) {
          setRows(newRows);
        }
      } catch (e) {}
    }
  };

  const updateRow = (id, field, value) => {
    setRows(rows.map(r => r.id === id ? { ...r, [field]: value } : r));
  };

  const deleteRow = (id) => {
    setRows(rows.filter(r => r.id !== id));
  };

  const addRow = () => {
    setRows([...rows, { id: Date.now(), appliance: "New Item", qty: 1, watts: 100, hours: 2 }]);
  };

  const totalDailyKwh = rows.reduce((acc, row) => acc + (row.qty * row.watts * row.hours) / 1000, 0);
  const totalCycleKwh = totalDailyKwh * billingDays;

  const calculateTNBill = (units) => {
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: 'var(--accent-primary)' }}>Precise Bill Calculator</h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn" style={{ background: 'var(--panel-bg)', color: 'white', border: '1px solid var(--border-color)' }} onClick={() => setRows(DEFAULT_ROWS)}>
            ↺ Reset
          </button>
          <button className="btn" style={{ background: 'var(--accent-secondary)', color: 'white', border: 'none' }} onClick={importMLPredictions}>
            📥 Import ML Predictions
          </button>
        </div>
      </div>

      <div className="form-group" style={{ maxWidth: '300px' }}>
        <label>Billing Cycle (Days)</label>
        <input 
          type="number" 
          className="form-control"
          min="1" max="365" 
          value={billingDays} 
          onChange={e => setBillingDays(parseInt(e.target.value))}
        />
      </div>

      <div style={{ overflowX: 'auto', marginTop: '20px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
              <th style={{ padding: '12px' }}>Appliance</th>
              <th style={{ padding: '12px' }}>Quantity</th>
              <th style={{ padding: '12px' }}>Wattage (W)</th>
              <th style={{ padding: '12px' }}>Hours/Day</th>
              <th style={{ padding: '12px' }}>Daily kWh</th>
              <th style={{ padding: '12px' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => {
              const dailyKwh = (row.qty * row.watts * row.hours) / 1000;
              return (
                <tr key={row.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '8px' }}>
                    <input className="form-control" value={row.appliance} onChange={e => updateRow(row.id, 'appliance', e.target.value)} />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input type="number" className="form-control" value={row.qty} onChange={e => updateRow(row.id, 'qty', parseFloat(e.target.value))} />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input type="number" className="form-control" value={row.watts} onChange={e => updateRow(row.id, 'watts', parseFloat(e.target.value))} />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input type="number" className="form-control" value={row.hours} onChange={e => updateRow(row.id, 'hours', parseFloat(e.target.value))} />
                  </td>
                  <td style={{ padding: '8px', color: 'var(--accent-primary)', fontWeight: 'bold' }}>
                    {dailyKwh.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px' }}>
                    <button onClick={() => deleteRow(row.id)} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '1.2rem' }}>×</button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <button onClick={addRow} className="btn" style={{ marginTop: '15px', background: 'rgba(255,255,255,0.1)', color: 'white' }}>
        + Add Row
      </button>

      <div className="metrics-container" style={{ marginTop: '40px' }}>
        <div className="metric-card glass-panel" style={{ background: 'var(--panel-bg)' }}>
          <div className="metric-label">Daily Consumption</div>
          <div className="metric-value" style={{ color: 'var(--accent-primary)' }}>
            {totalDailyKwh.toFixed(2)} <span className="metric-unit">kWh/day</span>
          </div>
        </div>
        <div className="metric-card warning glass-panel">
          <div className="metric-label">Total Cycle Units ({billingDays} days)</div>
          <div className="metric-value">
            {totalCycleKwh.toFixed(2)} <span className="metric-unit">units</span>
          </div>
        </div>
        <div className="metric-card success glass-panel" style={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}>
          <div className="metric-label" style={{ color: 'white' }}>Estimated Bill</div>
          <div className="metric-value" style={{ color: 'white' }}>
            ₹{calculateTNBill(totalCycleKwh)}
          </div>
        </div>
      </div>
    </div>
  );
}
