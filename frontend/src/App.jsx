import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import ResultPage from './pages/ResultPage'
import HistoryPage from './pages/HistoryPage'
import ModelsPage from './pages/ModelsPage'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50">
        <Navbar />
        <main className="max-w-5xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/"        element={<Home />} />
            <Route path="/result"  element={<ResultPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/models"  element={<ModelsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

