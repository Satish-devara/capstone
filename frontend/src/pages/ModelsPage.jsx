// pages/ModelsPage.jsx
import ModelComparison from '../components/ModelComparison'

export default function ModelsPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Model Comparison</h1>
        <p className="text-slate-500 text-sm mt-1">
          Evaluation metrics for all 5 ML models trained on the Pima Indians Diabetes Dataset.
          Metrics are computed using Stratified 10-Fold Cross-Validation.
        </p>
      </div>

      {/* Algorithm info cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        {[
          { name: 'XGBoost',    tag: 'Primary Model', emoji: '🏆',
            desc: 'Gradient boosting — highest ROC-AUC. Natively integrates with SHAP TreeExplainer.' },
          { name: 'Random Forest', tag: 'Ensemble',  emoji: '🌳',
            desc: 'Bagging of decision trees. Robust to noise and handles missing values well.' },
          { name: 'Logistic Regression', tag: 'Baseline', emoji: '📉',
            desc: 'Interpretable linear model. Provides baseline and direct coefficient insights.' },
        ].map(m => (
          <div key={m.name} className="card">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">{m.emoji}</span>
              <div>
                <p className="font-semibold text-slate-800 text-sm">{m.name}</p>
                <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{m.tag}</span>
              </div>
            </div>
            <p className="text-xs text-slate-500">{m.desc}</p>
          </div>
        ))}
      </div>

      <div className="card">
        <ModelComparison />
      </div>
    </div>
  )
}

