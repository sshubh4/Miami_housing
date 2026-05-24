import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const FEATURE_LABELS = {
  TOT_LVG_AREA:       'Living Area',
  OCEAN_DIST:         'Ocean Distance',
  LND_SQFOOT:         'Land Area',
  CNTR_DIST:          'CBD Distance',
  age:                'Property Age',
  structure_quality:  'Structure Quality',
  SPEC_FEAT_VAL:      'Special Features',
  log_ocean_dist:     'log(Ocean Dist)',
  log_cntr_dist:      'log(CBD Dist)',
  RAIL_DIST:          'Rail Distance',
  SUBCNTR_DI:         'Subcenter Dist',
  HWY_DIST:           'Highway Distance',
  WATER_DIST:         'Water Distance',
  cbd_access:         'CBD Access Score',
  land_to_living:     'Land/Living Ratio',
  is_coastal:         'Coastal Premium',
  peak_season:        'Peak Season',
  noise_penalty:      'Noise Penalty',
  avno60plus:         'Aircraft Noise',
  month_sold:         'Month Sold',
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="glass" style={{ padding: '8px 12px', fontSize: 11 }}>
      <div style={{ color: 'var(--accent)', fontWeight: 600, marginBottom: 2 }}>
        {FEATURE_LABELS[d.feature] || d.feature}
      </div>
      <div className="mono" style={{ color: 'var(--text)' }}>
        SHAP: {d.value.toFixed(4)}
      </div>
    </div>
  )
}

export default function ShapDrawer({ open, onToggle, shapData }) {
  const data = (shapData || [])
    .slice(0, 12)
    .map(d => ({ ...d, label: FEATURE_LABELS[d.feature] || d.feature }))
    .reverse()

  const maxVal = Math.max(...data.map(d => d.value), 0.001)

  return (
    <>
      {/* Toggle tab */}
      <button
        onClick={onToggle}
        className={`shap-tab-btn${open ? ' open' : ''}`}
        style={{
          position: 'fixed',
          bottom: open ? 296 : 0,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 500,
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderBottom: open ? '1px solid var(--border)' : 'none',
          borderRadius: open ? '8px 8px 0 0' : '8px 8px 0 0',
          padding: '6px 24px',
          color: open ? 'var(--accent)' : 'var(--text-muted)',
          fontSize: 10,
          fontFamily: 'var(--mono)',
          letterSpacing: '0.1em',
          cursor: 'pointer',
          transition: 'bottom 0.35s cubic-bezier(0.16,1,0.3,1), color 0.2s',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          backdropFilter: 'var(--blur)',
          WebkitBackdropFilter: 'var(--blur)',
        }}
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none"
          style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s' }}>
          <path d="M2 6l3-3 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        SHAP IMPORTANCE (Why this price?)
      </button>

      {/* Drawer */}
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: 296,
          zIndex: 450,
          transform: open ? 'translateY(0)' : 'translateY(100%)',
          transition: 'transform 0.35s cubic-bezier(0.16,1,0.3,1)',
          background: 'rgba(10,10,12,0.94)',
          backdropFilter: 'var(--blur)',
          WebkitBackdropFilter: 'var(--blur)',
          borderTop: '1px solid var(--border)',
          padding: '20px 24px 16px',
        }}
      >
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          marginBottom: 16,
        }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: 'var(--accent)' }}>
              SHAP FEATURE IMPORTANCE
            </div>
            <div className="muted" style={{ marginTop: 3 }}>
              Mean |SHAP| across 500 samples — what actually drives predictions
            </div>
          </div>
        </div>

        {data.length === 0 ? (
          <div style={{ display: 'flex', gap: 12, flexDirection: 'column', paddingTop: 8 }}>
            {[...Array(5)].map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 16, width: `${80 - i * 10}%` }} />
            ))}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 130 }}>
              <XAxis
                type="number"
                tick={{ fill: '#444', fontSize: 10, fontFamily: 'var(--mono)' }}
                axisLine={{ stroke: '#222' }}
                tickLine={false}
                domain={[0, maxVal * 1.1]}
                tickFormatter={v => v.toFixed(3)}
              />
              <YAxis
                type="category"
                dataKey="label"
                tick={{ fill: '#888', fontSize: 11, fontFamily: 'var(--sans)' }}
                axisLine={false}
                tickLine={false}
                width={130}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,255,135,0.05)' }} />
              <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={14}>
                {data.map((d, i) => {
                  const t = d.value / maxVal
                  const color = t > 0.7 ? '#00ff87' : t > 0.4 ? '#4af2a1' : t > 0.2 ? '#4a9eff' : '#2d5a8e'
                  return <Cell key={i} fill={color} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </>
  )
}
