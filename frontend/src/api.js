// BASE is empty in local dev (Vite proxy handles /api → :8000) and in Docker
// (nginx proxy_pass handles it). Set VITE_API_URL on Railway so the static
// frontend can reach the separate backend service directly.
const BASE = import.meta.env.VITE_API_URL || ''

export async function fetchMapData() {
  const r = await fetch(`${BASE}/api/data`)
  if (!r.ok) throw new Error(`/api/data ${r.status}`)
  return r.json()
}

export async function fetchNeighbourhoodStats() {
  const r = await fetch(`${BASE}/api/neighbourhood-stats`)
  if (!r.ok) throw new Error(`/api/neighbourhood-stats ${r.status}`)
  return r.json()
}

export async function predictPrice(params) {
  const r = await fetch(`${BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    // FastAPI validation errors return detail as an array of {loc, msg, type} objects
    const detail = Array.isArray(err.detail)
      ? err.detail.map(d => d.msg || String(d)).join('; ')
      : err.detail
    throw new Error(detail || `Prediction failed (${r.status})`)
  }
  return r.json()
}
