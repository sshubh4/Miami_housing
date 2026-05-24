function Chip({ label, value, loading, secondary, extraClass, title }) {
  const cls = ['stats-chip', secondary && 'stats-chip-secondary', extraClass].filter(Boolean).join(' ')
  return (
    <div
      className={cls}
      title={title}
      style={{ cursor: title ? 'help' : 'default',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        padding: '0 14px',
        borderRight: '1px solid var(--border)',
        flexShrink: 0,
      }}
    >
      <div className="label" style={{ marginBottom: 2 }}>{label}</div>
      {loading ? (
        <div className="skeleton" style={{ height: 16, width: 52 }} />
      ) : (
        <div className="mono" style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>
          {value}
        </div>
      )}
    </div>
  )
}

export default function StatsBar({ metrics, total, medianPrice, loading }) {
  return (
    <div
      className="glass stats-bar"
      style={{
        position: 'fixed',
        top: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 400,
        display: 'flex',
        alignItems: 'center',
        padding: '8px 0 8px 14px',
        whiteSpace: 'nowrap',
        animation: 'fadeIn 0.6s ease',
        maxWidth: 'calc(100vw - 32px)',
        overflow: 'hidden',
      }}
    >
      {/* Logo */}
      <div style={{ paddingRight: 14, borderRight: '1px solid var(--border)', flexShrink: 0 }}>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600,
          color: 'var(--accent)', letterSpacing: '0.03em',
        }}>
          Miami Housing Price
        </span>
      </div>

      <Chip label="Dataset"      value={total ? `${total.toLocaleString()} sales` : '—'} loading={loading} extraClass="stats-chip-dataset" />
      <Chip label="Model R²"     value={metrics ? metrics.r2.toFixed(3) : '—'}           loading={loading} title="R-squared: % of price variance explained by the model" />
      <Chip label="CV R²"        value={metrics?.cv_r2_mean ? metrics.cv_r2_mean.toFixed(3) : '—'} loading={loading} secondary title="Cross-validated R²: accuracy across 5 data splits" />
      <Chip label="MAE"          value={metrics ? `$${(metrics.mae / 1000).toFixed(1)}k` : '—'} loading={loading} secondary title="Mean Absolute Error: average prediction error in dollars" />
      <Chip label="Median Price" value={medianPrice ? `$${(medianPrice / 1000).toFixed(0)}k` : '—'} loading={loading} secondary />

      <div style={{ paddingLeft: 12, paddingRight: 6, flexShrink: 0 }}>
        <span className="label" style={{ color: 'var(--text-dim)', fontSize: 9 }}>XGBoost</span>
      </div>
    </div>
  )
}
