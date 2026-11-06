// components/Navbar.jsx
import { Link, useLocation } from 'react-router-dom'

const links = [
  { to: '/',        label: 'Predict' },
  { to: '/history', label: 'History' },
  { to: '/models',  label: 'Models' },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-2">
          <span className="text-2xl">🩺</span>
          <div>
            <p className="font-bold text-slate-800 leading-none">DiabetesXAI</p>
            <p className="text-xs text-slate-500 leading-none">Risk Prediction System</p>
          </div>
        </Link>

        {/* Navigation Links */}
        <ul className="flex items-center gap-1">
          {links.map(({ to, label }) => (
            <li key={to}>
              <Link
                to={to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  pathname === to
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}

