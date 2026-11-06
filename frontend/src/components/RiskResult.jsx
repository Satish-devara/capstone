// components/RiskResult.jsx
// Displays the prediction outcome: risk badge, probability gauge, and recommendation.

export default function RiskResult({ result }) {
  const { prediction, probability, risk_label, risk_category, recommendation, model_used } = result

  const categoryStyles = {
    low:      { bg: 'bg-green-50',  border: 'border-green-200',  text: 'text-green-700',  badge: 'badge-low',      icon: '✅' },
    moderate: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'badge-moderate', icon: '⚠️' },
    high:     { bg: 'bg-red-50',    border: 'border-red-200',    text: 'text-red-700',    badge: 'badge-high',     icon: '🚨' },
  }

  const style = categoryStyles[risk_category] || categoryStyles.low

  // Colour for the probability arc
  const arcColour = { low: '#16a34a', moderate: '#d97706', high: '#dc2626' }[risk_category]

  return (
    <div className={`${style.bg} border ${style.border} rounded-2xl p-6`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xl">{style.icon}</span>
            <h2 className="text-xl font-bold text-slate-800">Prediction Result</h2>
          </div>
          <span className={style.badge}>{risk_label}</span>
          <p className="text-xs text-slate-500 mt-2">Model used: {model_used}</p>
        </div>

        {/* Circular probability gauge */}
        <ProbabilityGauge value={probability} colour={arcColour} />
      </div>

      {/* Recommendation */}
      <div className="mt-5 p-4 bg-white rounded-xl border border-slate-100">
        <p className="text-sm font-semibold text-slate-700 mb-1">Recommendation</p>
        <p className="text-sm text-slate-600 leading-relaxed">{recommendation}</p>
      </div>

      <p className="text-xs text-slate-400 mt-4 text-center">
        This prediction is for awareness only and is NOT a medical diagnosis.
      </p>
    </div>
  )
}

// SVG circular gauge for probability
function ProbabilityGauge({ value, colour }) {
  const r  = 40
  const cx = 56
  const cy = 56
  const circumference = 2 * Math.PI * r
  const offset = circumference - (value / 100) * circumference

  return (
    <div className="flex flex-col items-center">
      <svg width="112" height="112" viewBox="0 0 112 112">
        {/* Background track */}
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="#e2e8f0" strokeWidth="10" />
        {/* Progress arc */}
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke={colour} strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: 'stroke-dashoffset 0.8s ease' }}
        />
        <text x={cx} y={cy + 6} textAnchor="middle"
          fontSize="18" fontWeight="700" fill={colour}>
          {value}%
        </text>
      </svg>
      <p className="text-xs text-slate-500 -mt-1">Risk Probability</p>
    </div>
  )
}

