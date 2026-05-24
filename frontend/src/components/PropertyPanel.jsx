import { useState, useEffect } from 'react'

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function useIsMobile() {
  const [mobile, setMobile] = useState(() => window.innerWidth < 768)   // <768 = mobile
  useEffect(() => {
    const fn = () => setMobile(window.innerWidth < 768)
    window.addEventListener('resize', fn)
    return () => window.removeEventListener('resize', fn)
  }, [])
  return mobile
}

export default function PropertyPanel({ form, onChange, metrics, loading }) {
  const [collapsed, setCollapsed] = useState(false)
  const isMobile = useIsMobile()

  function set(key, val) { onChange(key, val) }

  const panelStyle = isMobile ? {
    position: 'fixed',
    bottom: 0,
    left: 0,
    right: 0,
    width: '100%',
    borderRadius: '16px 16px 0 0',
    maxHeight: collapsed ? 56 : '72vh',
    overflowY: collapsed ? 'hidden' : 'auto',
    zIndex: 410,
    padding: collapsed ? '14px 20px' : '12px 20px 28px',
    transition: 'max-height 0.35s cubic-bezier(0.16,1,0.3,1), padding 0.2s',
  } : {
    position: 'fixed',
    top: 16,
    left: 16,
    width: 272,
    zIndex: 400,
    padding: collapsed ? '12px 16px' : '16px',
    transition: 'all 0.25s ease',
    animation: 'fadeIn 0.4s ease',
  }

  return (
    <div
      className="glass"
      style={panelStyle}
    >
      {/* Mobile drag handle */}
      {isMobile && (
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 10 }}>
          <div style={{ width: 36, height: 4, borderRadius: 2, background: 'var(--border)' }} />
        </div>
      )}

      {/* Header */}
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: collapsed ? 0 : 14, cursor: 'pointer',
        }}
        onClick={() => setCollapsed(c => !c)}
      >
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: 'var(--accent)' }}>
            PROPERTY INPUTS
          </div>
          {collapsed && (
            <div className="muted" style={{ marginTop: 2 }}>Click to expand</div>
          )}
        </div>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"
          style={{ color: 'var(--text-muted)', transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
          <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>

      {!collapsed && (
        <>
          <div className="divider" style={{ marginTop: 0 }} />

          {/* Living area */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <span className="label">Living Area</span>
              <span className="mono" style={{ fontSize: 12, color: 'var(--accent)' }}>
                {form.living_area.toLocaleString()} sqft
              </span>
            </div>
            <input type="range" min={400} max={10000} step={100} value={form.living_area}
              onChange={e => set('living_area', +e.target.value)} />
          </div>

          {/* Land area */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <span className="label">Land Area</span>
              <span className="mono" style={{ fontSize: 12, color: 'var(--accent)' }}>
                {form.land_area.toLocaleString()} sqft
              </span>
            </div>
            <input type="range" min={1000} max={60000} step={500} value={form.land_area}
              onChange={e => set('land_area', +e.target.value)} />
          </div>

          {/* Age */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <span className="label">Property Age</span>
              <span className="mono" style={{ fontSize: 12, color: 'var(--accent)' }}>
                {form.age} yr
              </span>
            </div>
            <input type="range" min={0} max={100} step={1} value={form.age}
              onChange={e => set('age', +e.target.value)} />
          </div>

          {/* Structure quality */}
          <div style={{ marginBottom: 14 }}>
            <div className="label" style={{ marginBottom: 7 }}>Structure Quality</div>
            <div className="quality-grid">
              {[1,2,3,4,5].map(q => (
                <button key={q} className={`quality-btn${form.structure_quality === q ? ' active' : ''}`}
                  onClick={() => set('structure_quality', q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Month sold */}
          <div style={{ marginBottom: 14 }}>
            <div className="label" style={{ marginBottom: 7 }}>Month Sold</div>
            <select value={form.month_sold} onChange={e => set('month_sold', +e.target.value)}>
              {MONTHS.map((m, i) => (
                <option key={i} value={i + 1}>{m}</option>
              ))}
            </select>
          </div>

          {/* Aircraft noise */}
          <div style={{ marginBottom: 12 }}>
            <label className="toggle-wrap" onClick={() => set('aircraft_noise', !form.aircraft_noise)}>
              <div className={`toggle-track${form.aircraft_noise ? ' on' : ''}`}>
                <div className="toggle-thumb" />
              </div>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Aircraft Noise Zone</span>
            </label>
          </div>

          <div className="divider" />

          {/* Model stats */}
          {loading ? (
            <div className="skeleton" style={{ height: 32, borderRadius: 6 }} />
          ) : metrics ? (
            <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <span style={{ color: 'var(--accent)' }}>XGBoost</span>
              {' '}· R² {metrics.r2}
              {' '}· MAE ${(metrics.mae / 1000).toFixed(1)}k
              {' '}· CV {metrics.cv_r2_mean?.toFixed(3)}
            </div>
          ) : null}

          {/* Glossary */}
          <div style={{
            marginTop: 12,
            paddingTop: 10,
            borderTop: '1px solid var(--border)',
            fontSize: 9,
            lineHeight: 1.9,
            color: 'rgba(0,255,135,0.4)',
            fontFamily: 'var(--mono)',
          }}>
            <div>SHAP = model explainability score</div>
            <div>CBD = Central Business District</div>
            <div>CV R² = cross-validated accuracy</div>
            <div>MAE = mean prediction error</div>
          </div>
        </>
      )}
    </div>
  )
}
