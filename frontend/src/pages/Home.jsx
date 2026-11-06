// pages/Home.jsx
import InputForm from '../components/InputForm'

export default function Home() {
  return (
    <div className="max-w-3xl mx-auto">
      {/* Hero */}
      <div className="text-center mb-8">
        <span className="text-5xl">🩺</span>
        <h1 className="mt-3 text-3xl font-bold text-slate-800">
          Diabetes Risk Prediction
        </h1>
        <p className="mt-2 text-slate-500 max-w-xl mx-auto">
          Enter your health parameters below. Our XAI system will predict your
          diabetes risk using <strong>XGBoost</strong> and explain the result
          using <strong>SHAP values</strong>.
        </p>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-2 mt-4">
          {['5 ML Models', 'SHAP Explainability', 'Risk Categorisation', 'Prediction History'].map(tag => (
            <span key={tag}
              className="bg-blue-50 text-blue-700 text-xs font-medium px-3 py-1 rounded-full border border-blue-100">
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Form card */}
      <div className="card">
        <InputForm />
      </div>
    </div>
  )
}

