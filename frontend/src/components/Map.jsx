import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet.heat'

delete L.Icon.Default.prototype._getIconUrl

const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
const TILE_ATTR =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ' +
  '&copy; <a href="https://carto.com/">CARTO</a>'

function priceColor(price) {
  if (price < 250_000) return '#4a9eff'
  if (price < 500_000) return '#00ff87'
  if (price < 800_000) return '#ffb347'
  return '#ff4757'
}

const NBHD_COLORS = ['#00ff87', '#4af2a1', '#ffb347', '#ff8c42', '#ff4757', '#c44569']

export default function Map({
  points, neighbourhoodStats, showChoropleth, showHeatmap,
  onMapClick, clickedLatLng,
}) {
  const containerRef = useRef(null)  // attached to the DOM div
  const leafletRef   = useRef(null)  // Leaflet map instance
  const heatRef      = useRef(null)
  const markersRef   = useRef(null)
  const choroplethRef = useRef(null)
  const clickMarkerRef = useRef(null)

  // ── Init — runs once, cleanup destroys the map so StrictMode re-creates cleanly ─
  useEffect(() => {
    const container = containerRef.current
    if (!container || leafletRef.current) return

    const map = L.map(container, {
      center: [25.65, -80.35],
      zoom: 10,
      minZoom: 7,
      maxZoom: 19,
      zoomControl: false,
      // Wide bounds — users can zoom out to see Florida / US east coast for context
      maxBounds: [[20.0, -90.0], [35.0, -70.0]],
      maxBoundsViscosity: 1.0,
    })

    L.tileLayer(TILE_URL, {
      attribution: TILE_ATTR,
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'bottomleft' }).addTo(map)

    leafletRef.current = map

    return () => {
      map.remove()
      leafletRef.current  = null
      heatRef.current     = null
      markersRef.current  = null
      choroplethRef.current   = null
      clickMarkerRef.current  = null
    }
  }, [])  // intentionally empty — init runs exactly once per mount

  // ── Click handler — re-registers when onMapClick identity changes ─────────
  useEffect(() => {
    const map = leafletRef.current
    if (!map) return
    const handler = (e) => onMapClick(e.latlng)
    map.on('click', handler)
    return () => { map.off('click', handler) }
  }, [onMapClick])

  // ── Heat layer ────────────────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current
    if (!map) return
    if (heatRef.current) { map.removeLayer(heatRef.current); heatRef.current = null }
    if (!showHeatmap || !points.length) return

    const maxPpsf = Math.max(...points.map(p => p.price_per_sqft || 0), 1)
    const data = points
      .filter(p => p.LATITUDE && p.LONGITUDE && p.price_per_sqft)
      .map(p => [p.LATITUDE, p.LONGITUDE, Math.min(p.price_per_sqft / maxPpsf, 1)])

    heatRef.current = L.heatLayer(data, {
      radius: 22,
      blur: 28,
      gradient: { 0.0: '#0a1628', 0.3: '#0d2d4a', 0.55: '#00ff87', 0.8: '#ffb347', 1.0: '#ff4757' },
      minOpacity: 0.35,
    }).addTo(map)
  }, [points, showHeatmap])

  // ── Property markers ──────────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current
    if (!map) return
    if (markersRef.current) { map.removeLayer(markersRef.current); markersRef.current = null }
    if (!points.length) return

    const layer = L.layerGroup()
    points.forEach(p => {
      if (!p.LATITUDE || !p.LONGITUDE) return
      L.circleMarker([p.LATITUDE, p.LONGITUDE], {
        radius: 3.5,
        color: 'transparent',
        fillColor: priceColor(p.SALE_PRC || 0),
        fillOpacity: 0.75,
        weight: 0,
      }).bindTooltip(
        `<span style="color:#00ff87;font-weight:600">$${((p.SALE_PRC||0)/1000).toFixed(0)}k</span><br>` +
        `${p.TOT_LVG_AREA ? Math.round(p.TOT_LVG_AREA).toLocaleString() : '—'} sqft &nbsp;·&nbsp; ` +
        `Age ${p.age ? Math.round(p.age) : '—'}y<br>` +
        `<span style="color:#666">${p.neighbourhood || ''}</span>`,
        { sticky: true, offset: [8, 0] },
      ).addTo(layer)
    })
    layer.addTo(map)
    markersRef.current = layer
  }, [points])

  // ── Choropleth ────────────────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current
    if (!map) return
    if (choroplethRef.current) { map.removeLayer(choroplethRef.current); choroplethRef.current = null }
    if (!showChoropleth || !neighbourhoodStats.length) return

    const maxPrice = Math.max(...neighbourhoodStats.map(n => n.median_price))
    const layer = L.layerGroup()
    neighbourhoodStats.forEach((n, i) => {
      const color = NBHD_COLORS[Math.min(i, NBHD_COLORS.length - 1)]
      const [[s, w], [n2, e]] = n.bounds
      L.rectangle([[s, w], [n2, e]], {
        color, weight: 1.5, fillColor: color,
        fillOpacity: 0.12, dashArray: '4 4',
      }).bindTooltip(
        `<span style="color:${color};font-weight:600">${n.neighbourhood}</span><br>` +
        `Median <span style="color:#00ff87">$${(n.median_price/1000).toFixed(0)}k</span><br>` +
        `<span style="color:#666">${n.count.toLocaleString()} sales</span>`,
        { sticky: true },
      ).addTo(layer)

      L.circleMarker([n.centroid_lat, n.centroid_lon], {
        radius: 6 + Math.round((n.median_price / maxPrice) * 8),
        color, fillColor: color, fillOpacity: 0.9, weight: 1.5,
      }).addTo(layer)
    })
    layer.addTo(map)
    choroplethRef.current = layer
  }, [showChoropleth, neighbourhoodStats])

  // ── Click pin ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current
    if (!map) return
    if (clickMarkerRef.current) { map.removeLayer(clickMarkerRef.current); clickMarkerRef.current = null }
    if (!clickedLatLng) return

    const dot = L.circleMarker([clickedLatLng.lat, clickedLatLng.lng], {
      radius: 8, color: '#00ff87', fillColor: '#00ff87',
      fillOpacity: 0.25, weight: 2,
    }).addTo(map)

    const ring = L.circleMarker([clickedLatLng.lat, clickedLatLng.lng], {
      radius: 16, color: 'rgba(0,255,135,0.3)',
      fillColor: 'transparent', weight: 1,
    }).addTo(map)

    clickMarkerRef.current = dot
    return () => { map.removeLayer(ring) }
  }, [clickedLatLng])

  return (
    <div
      ref={containerRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0 }}
    />
  )
}
