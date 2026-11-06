// pages/ResultPage.jsx
// Displays full prediction result + SHAP chart after form submission.

import { useLocation, useNavigate } from 'react-router-dom'
import RiskResult from '../components/RiskResult'
import ShapChart  from '../components/ShapChart'

export default function ResultPage() {
  const location = useLocation()
  const navigate = useNavigate()

  const result = location.state?.result
  const inputs = location.state?.inputs

  // Guard: redirect if someone navigates here directly without data
  if (!result) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-500 mb-4">No prediction data found.</p>
        <button onClick={() => navigate('/')} className="btn-primary">
          ← Back to Prediction Form
        </button>
      </div>
    )
  }

  const { shap_values, top_factors } = result

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Back button */}
      <button onClick={() => navigate('/')} className="btn-secondary text-sm">
        ← New Prediction
      </button>

      {/* Risk result card */}
      <div className="card p-0 overflow-hidden">
        <div className="px-6 pt-6 pb-4 border-b border-slate-100">
          <h2 className="text-lg font-bold text-slate-800">Risk Assessment</h2>
        </div>
        <div className="p-6">
          <RiskResult result={result} />
        </div>
      </div>

      {/* Top contributing factors */}
      {top_factors?.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-slate-800 mb-3">Top Risk Factors</h3>
          <div className="flex flex-wrap gap-2">
            {top_factors.map(f => (
              <span key={f}
                className="bg-red-50 text-red-700 border border-red-200 text-sm px-3 py-1 rounded-full">
                📌 {f}
              </span>
            ))}
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Features with the highest positive SHAP contribution to this prediction.
          </p>
        </div>
      )}

      {/* SHAP chart */}
      <div className="card">
        <ShapChart shapValues={shap_values} />
      </div>

      {/* Input summary */}
      <div className="card">
        <h3 className="font-semibold text-slate-800 mb-3">Input Summary</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {inputs && Object.entries(inputs).map(([key, val]) => (
            <div key={key} className="bg-slate-50 rounded-lg p-3">
              <p className="text-xs text-slate-500 capitalize">{key.replace(/([A-Z])/g, ' $1')}</p>
              <p className="font-semibold text-slate-800">{val}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-700">
        <strong>⚠️ Disclaimer:</strong> This system is for educational and awareness purposes only.
        It is not a clinical diagnostic tool. Always consult a qualified healthcare professional
        for medical advice and formal diabetes screening (HbA1c, OGTT).
      </div>
    </div>
  )
}

