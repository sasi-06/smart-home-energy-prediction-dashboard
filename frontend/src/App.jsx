import React from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import Predictor from './components/Predictor'
import BillCalculator from './components/BillCalculator'
import { Zap, Calculator } from 'lucide-react'

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <header className="header animate-fade-in">
          <h1>⚡ Smart Home Energy Predictor</h1>
          <p>Machine learning system analyzing electricity consumption and forecasting usage.</p>
        </header>

        <nav className="nav animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <NavLink 
            to="/predictor" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Zap size={18} /> Predictor
          </NavLink>
          <NavLink 
            to="/calculator" 
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Calculator size={18} /> Bill Calculator
          </NavLink>
        </nav>

        <main className="animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/predictor" replace />} />
            <Route path="/predictor" element={<Predictor />} />
            <Route path="/calculator" element={<BillCalculator />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
