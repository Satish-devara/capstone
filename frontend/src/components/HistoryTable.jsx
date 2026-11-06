// components/HistoryTable.jsx
// Paginated table of past predictions.

import { useState, useEffect } from 'react'
import { getPredictionHistory, deleteHistoryRecord } from '../services/api'

export default function HistoryTable() {
  const [page,    setPage]    = useState(1)
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  function fetchData(p = page) {
    setLoading(true)
    getPredictionHistory(p, 10)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchData(page) }, [page])

  async function handleDelete(id) {
    if (!window.confirm('Delete this prediction record?')) return
    try {
      await deleteHistoryRecord(id)
      fetchData(page)
    } catch (e) {
      alert('Delete failed: ' + e.message)
    }
  }

  if (loading && !data) return <Skeleton />
  if (error) return (
    <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
      {error}
    </div>
  )

  const { predictions = [], total = 0, pages = 1 } = data || {}

  if (predictions.length === 0) return (
    <div className="text-center py-16 text-slate-400">
      <p className="text-4xl mb-3">📂</p>
      <p className="font-medium">No predictions yet</p>
      <p className="text-sm">Submit health parameters on the Predict page to get started.</p>
    </div>
  )

  const badgeClass = {
    low:      'badge-low',
    moderate: 'badge-moderate',
    high:     'badge-high',
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">{total} predictions total</p>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left">
              <th className="py-3 pr-3 font-semibold text-slate-700">#</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">Date & Time</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">Age</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">Glucose</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">BMI</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">Risk</th>
              <th className="py-3 pr-3 font-semibold text-slate-700">Probability</th>
              <th className="py-3 font-semibold text-slate-700">Action</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map(pred => (
              <tr key={pred.id} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                <td className="py-3 pr-3 text-slate-400 font-mono">{pred.id}</td>
                <td className="py-3 pr-3 text-slate-600">
                  {new Date(pred.timestamp).toLocaleString()}
                </td>
                <td className="py-3 pr-3">{pred.inputs.age}</td>
                <td className="py-3 pr-3">{pred.inputs.glucose}</td>
                <td className="py-3 pr-3">{pred.inputs.bmi}</td>
                <td className="py-3 pr-3">
                  <span className={badgeClass[pred.risk_category] || 'badge-low'}>
                    {pred.risk_label}
                  </span>
                </td>
                <td className="py-3 pr-3 font-semibold">{pred.probability}%</td>
                <td className="py-3">
                  <button
                    onClick={() => handleDelete(pred.id)}
                    className="text-xs text-red-500 hover:text-red-700 hover:underline"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="btn-secondary text-sm disabled:opacity-40"
          >
            ← Prev
          </button>
          <span className="text-sm text-slate-600">
            Page {page} of {pages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(pages, p + 1))}
            disabled={page === pages}
            className="btn-secondary text-sm disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-3 mt-4">
      {[...Array(5)].map((_, i) => <div key={i} className="h-12 bg-slate-100 rounded-lg"/>)}
    </div>
  )
}

