const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function Stat({ label, value, accent }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div className="label" style={{ marginBottom: 2 }}>{label}</div>
      <div className="mono" style={{ fontSize: 13, color: accent ? 'var(--accent)' : 'var(--text)' }}>
        {value}
      </div>
    </div>
  )
}

export default function PredictionCard({ prediction, latlng, loading, form }) {
  if (!latlng && !loading) return null

  return (
    <div
      className="glass prediction-card-wrap"
      style={{
        position: 'fixed',
        bottom: 24,
        right: 16,
        width: 256,
        zIndex: 400,
        padding: '16px',
        animation: 'slideUp 0.3s cubic-bezier(0.16,1,0.3,1)',
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: 'var(--accent)', marginBottom: 12 }}>
        PRICE ESTIMATE
      </div>

      {loading ? (
        <div>
          <div className="skeleton" style={{ height: 36, width: '80%', marginBottom: 12 }} />
          <div className="skeleton" style={{ height: 16, marginBottom: 8 }} />
          <div className="skeleton" style={{ height: 16, width: '60%' }} />
        </div>
      ) : prediction ? (
        <>
          {/* Main price */}
          <div style={{ marginBottom: 12 }}>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 28,
              fontWeight: 600,
              color: '#fff',
              letterSpacing: '-0.02em',
              lineHeight: 1,
            }}>
              ${prediction.predicted.toLocaleString()}
            </div>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'var(--text-muted)',
              marginTop: 5,
            }}>
              ${prediction.confidence_low.toLocaleString()} – ${prediction.confidence_high.toLocaleString()}
            </div>
          </div>

          <div className="divider" />

          {/* Location */}
          <div style={{ marginBottom: 10 }}>
            <Stat label="Coordinates"
              value={`${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}`} />
            <Stat label="Ocean Distance"
              value={`${(prediction.ocean_dist_m / 1000).toFixed(2)} km${prediction.is_coastal ? '  ◆ coastal' : ''}`}
              accent={prediction.is_coastal} />
            <Stat label="CBD Distance"
              value={`${(prediction.cntr_dist_m / 1000).toFixed(2)} km`} />
          </div>

          {/* Comparables */}
          {prediction.comparables?.length > 0 && (
            <>
              <div className="divider" />
              <div className="label" style={{ marginBottom: 8 }}>Nearby Sales</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {prediction.comparables.slice(0, 4).map((c, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '4px 8px',
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: 5,
                    border: '1px solid var(--border)',
                  }}>
                    <div>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--accent)' }}>
                        ${(c.sale_price / 1000).toFixed(0)}k
                      </div>
                      <div className="muted" style={{ fontSize: 10 }}>
                        {(c.distance_m / 1000).toFixed(2)} km away
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="mono" style={{ fontSize: 10, color: 'var(--text)' }}>
                        {c.sqft?.toLocaleString()} sf
                      </div>
                      <div className="muted" style={{ fontSize: 10 }}>
                        {c.age}y old
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      ) : (
        <div style={{ color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>
          Click anywhere on the map to predict the price at that location using your current property inputs.
        </div>
      )}
    </div>
  )
}
