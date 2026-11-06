// components/ModelComparison.jsx
// Table + radar chart comparing evaluation metrics across all 5 ML models.

import { useState, useEffect } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { getModelMetrics } from '../services/api'

const METRIC_LABELS = {
  accuracy:  'Accuracy',
  precision: 'Precision',
  recall:    'Recall',
  f1_score:  'F1-Score',
  roc_auc:   'ROC-AUC',
}

const MODEL_COLORS = {
  LogisticRegression: '#3b82f6',
  RandomForest:       '#10b981',
  XGBoost:            '#f59e0b',
  SVM:                '#8b5cf6',
  KNN:                '#ef4444',
}

export default function ModelComparison() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    getModelMetrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Skeleton />
  if (error)   return <ErrorBox message={error} />
  if (!metrics) return null

  const models      = Object.keys(metrics)
  const metricKeys  = Object.keys(METRIC_LABELS)

  // Find the best model per metric (for highlighting)
  const bestPerMetric = {}
  metricKeys.forEach(m => {
    bestPerMetric[m] = models.reduce((best, name) =>
      (metrics[name][m] || 0) > (metrics[best][m] || 0) ? name : best, models[0])
  })

  // Build radar chart data
  const radarData = metricKeys.map(key => {
    const entry = { metric: METRIC_LABELS[key] }
    models.forEach(name => {
      entry[name] = +(((metrics[name][key] || 0) * 100).toFixed(1))
    })
    return entry
  })

  return (
    <div className="space-y-8">
      {/* Metrics Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-3 pr-4 font-semibold text-slate-700">Model</th>
              {metricKeys.map(k => (
                <th key={k} className="text-center py-3 px-2 font-semibold text-slate-700">
                  {METRIC_LABELS[k]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map(name => (
              <tr key={name} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: MODEL_COLORS[name] }}
                    />
                    <span className="font-medium text-slate-800">
                      {name}
                      {name === 'XGBoost' && (
                        <span className="ml-2 text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                          Primary
                        </span>
                      )}
                    </span>
                  </div>
                </td>
                {metricKeys.map(k => {
                  const val = metrics[name][k]
                  const isBest = bestPerMetric[k] === name
                  return (
                    <td key={k} className={`text-center py-3 px-2 font-mono ${isBest ? 'font-bold text-blue-700' : 'text-slate-700'}`}>
                      {val !== undefined ? (val * 100).toFixed(1) + '%' : '—'}
                      {isBest && <span className="ml-1 text-xs">★</span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-slate-400 mt-2">★ Best value per metric | Evaluated with Stratified 10-Fold Cross-Validation</p>
      </div>

      {/* Radar Chart */}
      <div>
        <h4 className="font-semibold text-slate-700 mb-4">Metric Radar Comparison</h4>
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[60, 100]} tick={{ fontSize: 10 }} />
            {models.map(name => (
              <Radar
                key={name}
                name={name}
                dataKey={name}
                stroke={MODEL_COLORS[name]}
                fill={MODEL_COLORS[name]}
                fillOpacity={0.08}
                strokeWidth={2}
              />
            ))}
            <Tooltip formatter={v => `${v}%`} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="h-10 bg-slate-100 rounded-lg" />
      ))}
    </div>
  )
}

function ErrorBox({ message }) {
  return (
    <div className="p-4 bg-orange-50 border border-orange-200 rounded-xl text-sm text-orange-700">
      <strong>Models not ready:</strong> {message}
      <br />
      <code className="mt-1 block text-xs bg-orange-100 p-2 rounded">
        cd backend && python ml/train.py
      </code>
    </div>
  )
}

