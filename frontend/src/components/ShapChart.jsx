// components/ShapChart.jsx
// Horizontal bar chart visualising SHAP feature contributions.
// Positive SHAP = red (increases risk), Negative SHAP = green (decreases risk).

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'

const FEATURE_LABELS = {
  Pregnancies:              'Pregnancies',
  Glucose:                  'Glucose (mg/dL)',
  BloodPressure:            'Blood Pressure',
  SkinThickness:            'Skin Thickness',
  Insulin:                  'Insulin (μU/mL)',
  BMI:                      'BMI (kg/m²)',
  DiabetesPedigreeFunction: 'Diabetes Pedigree',
  Age:                      'Age (years)',
}

export default function ShapChart({ shapValues }) {
  if (!shapValues) return null

  // Convert to chart-friendly format, sorted by absolute value
  const data = Object.entries(shapValues)
    .map(([feature, value]) => ({
      feature: FEATURE_LABELS[feature] || feature,
      value:   value,
      absVal:  Math.abs(value),
    }))
    .sort((a, b) => b.absVal - a.absVal)

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const v = payload[0].value
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm">
        <p className="font-semibold text-slate-700">{payload[0].payload.feature}</p>
        <p className={v > 0 ? 'text-red-600' : 'text-green-600'}>
          SHAP: {v > 0 ? '+' : ''}{v.toFixed(4)}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          {v > 0 ? '↑ Increases diabetes risk' : '↓ Decreases diabetes risk'}
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-slate-800">Feature Contributions (SHAP)</h3>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-400 inline-block"/>
            Increases Risk
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-green-500 inline-block"/>
            Decreases Risk
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 30, bottom: 4, left: 140 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11 }}
            tickFormatter={v => v.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 12 }}
            width={135}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1.5} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.value >= 0 ? '#f87171' : '#4ade80'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-xs text-slate-400 text-center mt-2">
        SHAP (SHapley Additive exPlanations) — mathematically rigorous feature attribution using XGBoost TreeExplainer
      </p>
    </div>
  )
}

