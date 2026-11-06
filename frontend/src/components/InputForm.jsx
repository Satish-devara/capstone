// components/InputForm.jsx
// Health parameter input form with client-side validation.

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predictDiabetes } from '../services/api'

// Field definitions — label, key, min, max, unit, hint
const FIELDS = [
  {
    key: 'pregnancies',
    label: 'Number of Pregnancies',
    type: 'number', min: 0, max: 20, step: 1,
    unit: 'times',
    hint: 'Total number of pregnancies (enter 0 if male or nulliparous)',
    placeholder: 'e.g. 2',
  },
  {
    key: 'glucose',
    label: 'Plasma Glucose',
    type: 'number', min: 50, max: 300, step: 1,
    unit: 'mg/dL',
    hint: '2-hour oral glucose tolerance test concentration',
    placeholder: 'e.g. 120',
  },
  {
    key: 'bloodPressure',
    label: 'Diastolic Blood Pressure',
    type: 'number', min: 30, max: 180, step: 1,
    unit: 'mm Hg',
    hint: 'Lower number in a blood pressure reading',
    placeholder: 'e.g. 72',
  },
  {
    key: 'skinThickness',
    label: 'Skin Thickness',
    type: 'number', min: 0, max: 100, step: 1,
    unit: 'mm',
    hint: 'Triceps skinfold thickness (enter 0 if unknown)',
    placeholder: 'e.g. 20',
  },
  {
    key: 'insulin',
    label: '2-Hour Serum Insulin',
    type: 'number', min: 0, max: 900, step: 1,
    unit: 'μU/mL',
    hint: '2-hour insulin level (enter 0 if unknown)',
    placeholder: 'e.g. 80',
  },
  {
    key: 'bmi',
    label: 'Body Mass Index (BMI)',
    type: 'number', min: 10, max: 80, step: 0.1,
    unit: 'kg/m²',
    hint: 'Weight in kg divided by height in m²',
    placeholder: 'e.g. 28.5',
  },
  {
    key: 'diabetesPedigreeFunction',
    label: 'Diabetes Pedigree Function',
    type: 'number', min: 0, max: 3, step: 0.001,
    unit: 'score',
    hint: 'Family history likelihood score (0.0 – 3.0)',
    placeholder: 'e.g. 0.472',
  },
  {
    key: 'age',
    label: 'Age',
    type: 'number', min: 1, max: 120, step: 1,
    unit: 'years',
    hint: 'Patient age in years',
    placeholder: 'e.g. 35',
  },
]

const INITIAL_VALUES = Object.fromEntries(FIELDS.map(f => [f.key, '']))

export default function InputForm() {
  const navigate  = useNavigate()
  const [values,  setValues]  = useState(INITIAL_VALUES)
  const [errors,  setErrors]  = useState({})
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')

  // ── Client-side validation ─────────────────────────────────────────────────
  function validate() {
    const newErrors = {}
    FIELDS.forEach(({ key, min, max, hint, label }) => {
      const val = values[key]
      if (val === '' || val === null || val === undefined) {
        newErrors[key] = `${label} is required.`
      } else if (isNaN(Number(val))) {
        newErrors[key] = `${label} must be a number.`
      } else if (Number(val) < min || Number(val) > max) {
        newErrors[key] = `Must be between ${min} and ${max}.`
      }
    })
    return newErrors
  }

  function handleChange(e) {
    const { name, value } = e.target
    setValues(prev => ({ ...prev, [name]: value }))
    if (errors[name]) setErrors(prev => ({ ...prev, [name]: '' }))
    setApiError('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const validationErrors = validate()
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors)
      return
    }

    setLoading(true)
    setApiError('')
    try {
      // Convert all values to numbers before sending
      const payload = Object.fromEntries(
        Object.entries(values).map(([k, v]) => [k, Number(v)])
      )
      const result = await predictDiabetes(payload)
      navigate('/result', { state: { result, inputs: payload } })
    } catch (err) {
      setApiError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setValues(INITIAL_VALUES)
    setErrors({})
    setApiError('')
  }

  // ── Example data (for demo / teacher presentation) ─────────────────────────
  function fillExample(type) {
    if (type === 'high') {
      setValues({
        pregnancies: '8', glucose: '183', bloodPressure: '64',
        skinThickness: '0', insulin: '0', bmi: '23.3',
        diabetesPedigreeFunction: '0.672', age: '32',
      })
    } else {
      setValues({
        pregnancies: '1', glucose: '89', bloodPressure: '66',
        skinThickness: '23', insulin: '94', bmi: '28.1',
        diabetesPedigreeFunction: '0.167', age: '21',
      })
    }
    setErrors({})
  }

  return (
    <form onSubmit={handleSubmit} noValidate>
      {/* Demo buttons */}
      <div className="flex gap-2 mb-6">
        <span className="text-sm text-slate-500 self-center mr-1">Quick fill:</span>
        <button type="button" onClick={() => fillExample('low')}
          className="btn-secondary text-sm">
          🟢 Low Risk Example
        </button>
        <button type="button" onClick={() => fillExample('high')}
          className="btn-secondary text-sm">
          🔴 High Risk Example
        </button>
        <button type="button" onClick={handleReset}
          className="btn-secondary text-sm ml-auto">
          Reset
        </button>
      </div>

      {/* Input grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {FIELDS.map(({ key, label, type, min, max, step, unit, hint, placeholder }) => (
          <div key={key}>
            <label htmlFor={key} className="label">
              {label}
              <span className="ml-1 text-xs text-slate-400 font-normal">({unit})</span>
            </label>
            <input
              id={key}
              name={key}
              type={type}
              min={min}
              max={max}
              step={step}
              value={values[key]}
              onChange={handleChange}
              placeholder={placeholder}
              className={`input-field ${errors[key] ? 'border-red-400 ring-1 ring-red-400' : ''}`}
            />
            {errors[key]
              ? <p className="mt-1 text-xs text-red-600">{errors[key]}</p>
              : <p className="mt-1 text-xs text-slate-400">{hint}</p>
            }
          </div>
        ))}
      </div>

      {/* API error */}
      {apiError && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          ⚠️ {apiError}
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="btn-primary w-full mt-6 flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z"/>
            </svg>
            Analysing with XAI...
          </>
        ) : (
          '🔍 Predict Diabetes Risk'
        )}
      </button>

      <p className="text-center text-xs text-slate-400 mt-3">
        ⚠️ For educational awareness only — not a medical diagnosis.
      </p>
    </form>
  )
}

