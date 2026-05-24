import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import Map from './components/Map.jsx'
import PropertyPanel from './components/PropertyPanel.jsx'
import PredictionCard from './components/PredictionCard.jsx'
import StatsBar from './components/StatsBar.jsx'

const ShapDrawer = lazy(() => import('./components/ShapDrawer.jsx'))
import { fetchMapData, fetchNeighbourhoodStats, predictPrice } from './api.js'

// Tight enough to exclude Everglades (west) and open ocean (east)
// while covering all developed Miami-Dade residential areas
const MIAMI_BOUNDS = { minLat: 25.1, maxLat: 26.0, minLon: -80.6, maxLon: -80.0 }

const DEFAULT_FORM = {
  living_area: 1800,
  land_area: 7500,
  age: 30,
  structure_quality: 3,
  month_sold: 3,
  aircraft_noise: false,
}

export default function App() {
  const [mapData, setMapData] = useState(null)
  const [nbhdStats, setNbhdStats] = useState([])
  const [dataLoading, setDataLoading] = useState(true)

  const [form, setForm] = useState(DEFAULT_FORM)
  const [clickedLatLng, setClickedLatLng] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState(null)

  const [infoMsg, setInfoMsg] = useState(null)
  const [showChoropleth, setShowChoropleth] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [drawerOpen, setDrawerOpen] = useState(false)

  // ── Load initial data ──────────────────────────────────────────────────────
  useEffect(() => {
    setDataLoading(true)
    Promise.all([fetchMapData(), fetchNeighbourhoodStats()])
      .then(([data, nbhd]) => {
        setMapData(data)
        setNbhdStats(nbhd)
      })
      .catch(console.error)
      .finally(() => setDataLoading(false))
  }, [])

  // ── Form field change ──────────────────────────────────────────────────────
  function handleFormChange(key, value) {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  // ── Map click → predict ────────────────────────────────────────────────────
  const handleMapClick = useCallback(async (latlng) => {
    const { lat, lng } = latlng
    // Bounds check BEFORE any API call — reject clicks outside Miami-Dade County
    if (
      lat < MIAMI_BOUNDS.minLat || lat > MIAMI_BOUNDS.maxLat ||
      lng < MIAMI_BOUNDS.minLon || lng > MIAMI_BOUNDS.maxLon
    ) {
      setInfoMsg('Click within Miami-Dade County to get a prediction')
      setTimeout(() => setInfoMsg(null), 3000)
      return
    }

    setClickedLatLng(latlng)
    setPrediction(null)
    setPredError(null)
    setPredLoading(true)
    try {
      const result = await predictPrice({
        lat: latlng.lat,
        lon: latlng.lng,
        ...form,
      })
      setPrediction(result)
    } catch (err) {
      setPredError(err.message)
    } finally {
      setPredLoading(false)
    }
  }, [form])

  return (
    <>
      {/* Full-viewport map — rendered outside React's paint cycle */}
      <Map
        points={mapData?.points ?? []}
        neighbourhoodStats={nbhdStats}
        showChoropleth={showChoropleth}
        showHeatmap={showHeatmap}
        onMapClick={handleMapClick}
        clickedLatLng={clickedLatLng}
      />

      {/* Top stats bar */}
      <StatsBar
        metrics={mapData?.metrics}
        total={mapData?.total}
        medianPrice={mapData?.median_price}
        loading={dataLoading}
      />

      {/* Layer controls — top right */}
      <div
        className="glass layers-panel"
        style={{
          position: 'fixed',
          top: 16,
          right: 16,
          zIndex: 400,
          padding: '10px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          animation: 'fadeIn 0.5s ease',
        }}
      >
        <div className="label" style={{ marginBottom: 2 }}>LAYERS</div>

        <div className="toggle-wrap" style={{ cursor: 'pointer' }}
          onClick={() => setShowHeatmap(v => !v)}>
          <div className={`toggle-track${showHeatmap ? ' on' : ''}`}>
            <div className="toggle-thumb" />
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Heat Map</span>
        </div>

        <div className="toggle-wrap" style={{ cursor: 'pointer' }}
          onClick={() => setShowChoropleth(v => !v)}>
          <div className={`toggle-track${showChoropleth ? ' on' : ''}`}>
            <div className="toggle-thumb" />
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Neighbourhoods</span>
        </div>
      </div>

      {/* Left property panel */}
      <PropertyPanel
        form={form}
        onChange={handleFormChange}
        metrics={mapData?.metrics}
        loading={dataLoading}
      />

      {/* Bottom-right prediction card */}
      <PredictionCard
        prediction={prediction}
        latlng={clickedLatLng}
        loading={predLoading}
        form={form}
      />

      {/* Error toast */}
      {predError && (
        <div style={{
          position: 'fixed',
          bottom: 80,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 600,
          background: 'rgba(255,71,87,0.15)',
          border: '1px solid rgba(255,71,87,0.4)',
          color: '#ff4757',
          padding: '8px 20px',
          borderRadius: 8,
          fontSize: 12,
          fontFamily: 'var(--mono)',
          animation: 'slideUp 0.25s ease',
        }}>
          {predError}
        </div>
      )}

      {/* Info toast — out-of-bounds click message */}
      {infoMsg && (
        <div style={{
          position: 'fixed',
          bottom: 80,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 600,
          background: 'rgba(74,158,255,0.15)',
          border: '1px solid rgba(74,158,255,0.4)',
          color: '#4a9eff',
          padding: '8px 20px',
          borderRadius: 8,
          fontSize: 12,
          fontFamily: 'var(--mono)',
          animation: 'slideUp 0.25s ease',
          whiteSpace: 'nowrap',
        }}>
          {infoMsg}
        </div>
      )}

      {/* SHAP drawer + toggle tab — Recharts loaded lazily on first open */}
      <Suspense fallback={null}>
        <ShapDrawer
          open={drawerOpen}
          onToggle={() => setDrawerOpen(v => !v)}
          shapData={mapData?.shap}
        />
      </Suspense>

      {/* Price legend — bottom left (hidden on mobile) */}
      <div
        className="glass legend-widget"
        style={{
          position: 'fixed',
          bottom: 56,
          left: 16,
          zIndex: 400,
          padding: '10px 14px',
          animation: 'fadeIn 0.7s ease',
        }}
      >
        <div className="label" style={{ marginBottom: 8 }}>SALE PRICE</div>
        {[
          { color: '#4a9eff', label: '< $250k' },
          { color: '#00ff87', label: '$250k – $500k' },
          { color: '#ffb347', label: '$500k – $800k' },
          { color: '#ff4757', label: '> $800k' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
            <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</span>
          </div>
        ))}
        <div style={{ height: 1, background: 'var(--border)', margin: '8px 0' }} />
        <div className="muted" style={{ fontSize: 10, fontStyle: 'italic' }}>Click map to predict</div>
      </div>
    </>
  )
}
