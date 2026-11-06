// pages/HistoryPage.jsx
import HistoryTable from '../components/HistoryTable'

export default function HistoryPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Prediction History</h1>
        <p className="text-slate-500 text-sm mt-1">
          All past diabetes risk predictions stored in the database.
        </p>
      </div>
      <div className="card">
        <HistoryTable />
      </div>
    </div>
  )
}

