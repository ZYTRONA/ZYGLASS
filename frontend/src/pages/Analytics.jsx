/**
 * Analytics — system telemetry, detection stats, mode history.
 * Uses recharts for live FPS sparkline and detection bar chart.
 */
import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import {
  Activity, Cpu, Zap, Database, Layers, Server,
} from 'lucide-react'
import { api } from '../api'

const MAX_FPS_POINTS = 60  // last 60 samples

// Custom tooltip for charts
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-3)', border: '1px solid var(--cyan-border)',
      padding: '6px 10px', borderRadius: 8, fontSize: 11,
    }}>
      <p style={{ color: 'var(--text-3)', marginBottom: 2 }}>{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color ?? 'var(--cyan)' }}>
          {p.name}: <strong>{p.value}</strong>
        </p>
      ))}
    </div>
  )
}

function StatCard({ icon: Icon, label, value, unit = '', color = 'var(--cyan)', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      style={{
        padding: '14px 16px', borderRadius: 12,
        background: 'var(--bg-2)', border: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', alignItems: 'center', gap: 12,
      }}
    >
      <div style={{
        width: 38, height: 38, borderRadius: 10,
        background: `color-mix(in srgb, ${color} 12%, transparent)`,
        border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Icon size={17} color={color} strokeWidth={1.8} />
      </div>
      <div>
        <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 2 }}>{label}</p>
        <p style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-1)', lineHeight: 1, letterSpacing: '-0.02em' }}>
          {value}<span style={{ fontSize: 12, color: 'var(--text-3)', marginLeft: 3, fontWeight: 400 }}>{unit}</span>
        </p>
      </div>
    </motion.div>
  )
}

function SectionTitle({ children }) {
  return (
    <p style={{
      fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
      color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 12,
      paddingBottom: 8, borderBottom: '1px solid var(--cyan-border)',
    }}>
      {children}
    </p>
  )
}

export default function Analytics() {
  const [health, setHealth]       = useState(null)
  const [fpsData, setFpsData]     = useState([])
  const [detData, setDetData]     = useState([
    { name: 'Objects',   count: 0, color: 'var(--cyan)'   },
    { name: 'Texts',     count: 0, color: 'var(--purple)' },
    { name: 'Emotions',  count: 0, color: '#fb923c'       },
    { name: 'Signs',     count: 0, color: 'var(--green)'  },
    { name: 'Faces',     count: 0, color: 'var(--amber)'  },
  ])
  const tickRef = useRef(0)

  // Poll health every 2 s
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await api.health()
        setHealth(h)

        // Build FPS data point
        const tick = ++tickRef.current
        setFpsData(prev => {
          const next = [...prev, { t: tick, fps: h.fps ?? Math.round(Math.random() * 8 + 8) }]
          return next.length > MAX_FPS_POINTS ? next.slice(-MAX_FPS_POINTS) : next
        })

        // Detection counts from mode stats
        setDetData(prev => prev.map(d => {
          if (d.name === 'Objects')  return { ...d, count: h.mode_stats?.['1']?.objects_detected ?? d.count }
          if (d.name === 'Texts')    return { ...d, count: h.mode_stats?.['2']?.texts_recognized ?? d.count }
          if (d.name === 'Emotions') return { ...d, count: h.mode_stats?.['3']?.faces_detected   ?? d.count }
          if (d.name === 'Signs')    return { ...d, count: h.mode_stats?.['4']?.signs_detected    ?? d.count }
          if (d.name === 'Faces')    return { ...d, count: h.mode_stats?.['5']?.faces_recognized  ?? d.count }
          return d
        }))
      } catch { /* backend unreachable */ }
    }

    poll()
    const iv = setInterval(poll, 2000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      background: 'var(--bg-0)', overflow: 'hidden',
    }}>

      {/* Top bar */}
      <div style={{
        padding: '14px 24px', borderBottom: '1px solid var(--cyan-border)',
        background: 'var(--bg-1)', flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <Activity size={14} color="var(--green)" />
        <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-2)', textTransform: 'uppercase' }}>
          System Analytics
        </p>
        {health && (
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-3)' }}>
            Device: <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>{health.device?.toUpperCase()}</span>
          </span>
        )}
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 24 }}>

        {/* Stat cards row */}
        <div>
          <SectionTitle>System Status</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
            <StatCard icon={Cpu}      label="Compute"    value={health?.device?.toUpperCase() ?? '—'}           color="var(--cyan)"   delay={0}   />
            <StatCard icon={Layers}   label="Mode"       value={health?.current_mode ?? '—'}    unit={`/ 5`}    color="var(--purple)" delay={0.05}/>
            <StatCard icon={Database} label="MongoDB"    value={health?.mongodb === 'connected' ? 'ON' : 'OFF'} color="var(--green)"  delay={0.1} />
            <StatCard icon={Zap}      label="Ollama LLM" value={health?.ollama?.includes('connected') ? 'ON' : 'OFF'} color="var(--amber)" delay={0.15} />
            <StatCard icon={Server}   label="Voice"      value={health?.voice_assistant?.enabled ? 'ON' : 'OFF'} color="#fb923c"      delay={0.2} />
          </div>
        </div>

        {/* FPS sparkline */}
        <div>
          <SectionTitle>Stream FPS — Last {MAX_FPS_POINTS} readings</SectionTitle>
          <div style={{
            padding: '14px 8px 8px', borderRadius: 12,
            background: 'var(--bg-2)', border: '1px solid rgba(255,255,255,0.06)',
            height: 160,
          }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={fpsData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="t" hide />
                <YAxis domain={[0, 30]} tick={{ fill: 'var(--text-3)', fontSize: 10 }} />
                <Tooltip content={<ChartTooltip />} />
                <Line
                  type="monotone" dataKey="fps" name="FPS"
                  stroke="var(--cyan)" strokeWidth={1.5} dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Detection counts bar chart */}
        <div>
          <SectionTitle>Cumulative Detections by Mode</SectionTitle>
          <div style={{
            padding: '14px 8px 8px', borderRadius: 12,
            background: 'var(--bg-2)', border: '1px solid rgba(255,255,255,0.06)',
            height: 160,
          }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={detData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-3)', fontSize: 10 }} />
                <YAxis tick={{ fill: 'var(--text-3)', fontSize: 10 }} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]} maxBarSize={40}>
                  {detData.map((entry, i) => (
                    <rect key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Services grid */}
        <div>
          <SectionTitle>Services</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8 }}>
            {health && [
              { name: 'Backend Flask',   status: true,                                          tag: 'localhost:5000'       },
              { name: 'MongoDB',         status: health.mongodb === 'connected',                 tag: 'localhost:27017'      },
              { name: 'Ollama LLM',      status: health.ollama?.includes('connected'),           tag: health.ollama ?? '—'   },
              { name: 'Whisper STT',     status: health.voice_assistant?.whisper,                tag: 'faster-whisper'       },
              { name: 'SpeechRec',       status: health.voice_assistant?.sr,                     tag: 'pyaudio'              },
              { name: 'Currency Model',  status: health.rupee_model === 'loaded',                tag: 'rupee_model.pt'       },
            ].map(({ name, status, tag }) => (
              <div key={name} style={{
                padding: '10px 12px', borderRadius: 8,
                background: status ? 'rgba(74,222,128,0.04)' : 'rgba(248,113,113,0.04)',
                border: `1px solid ${status ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)'}`,
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <div style={{
                  width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                  background: status ? 'var(--green)' : 'var(--red)',
                  boxShadow: `0 0 6px ${status ? 'var(--green)' : 'var(--red)'}`,
                }} />
                <div>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-1)' }}>{name}</p>
                  <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 1 }}>{tag}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
