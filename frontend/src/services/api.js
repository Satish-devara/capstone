// services/api.js
// Centralised Axios instance for all backend API calls.

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,   // 30s — SHAP computation can be slow on first call
})

// ── Response Interceptor ──────────────────────────────────────────────────────
// Unwrap data and normalise errors so callers always get { data } or throw Error.
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.response?.data?.message ||
      error.message ||
      'Network error. Please try again.'
    return Promise.reject(new Error(message))
  }
)

// ── API Methods ───────────────────────────────────────────────────────────────

/** Submit health parameters and get a diabetes risk prediction. */
export const predictDiabetes = (formData) =>
  api.post('/api/predict', formData)

/** Fetch paginated prediction history. */
export const getPredictionHistory = (page = 1, limit = 10) =>
  api.get(`/api/history?page=${page}&limit=${limit}`)

/** Fetch a single history record by ID. */
export const getHistoryRecord = (id) =>
  api.get(`/api/history/${id}`)

/** Delete a single history record. */
export const deleteHistoryRecord = (id) =>
  api.delete(`/api/history/${id}`)

/** Fetch evaluation metrics for all trained models. */
export const getModelMetrics = () =>
  api.get('/api/models')

/** Check backend health. */
export const checkHealth = () =>
  api.get('/api/health')

