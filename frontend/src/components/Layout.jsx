/**
 * Layout — persistent shell: sidebar nav + main content slot.
 * Animated active nav indicator via Framer Motion layoutId.
 */
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Grid3x3, BrainCircuit, BarChart2,
  Mic, MicOff, Eye, Signal, SignalZero, Power,
  ChevronRight,
} from 'lucide-react'

const NAV = [
  { id: 'dashboard',  label: 'Dashboard',   Icon: LayoutDashboard },
  { id: 'modes',      label: 'Modes',        Icon: Grid3x3         },
  { id: 'ai',         label: 'ZyGlass AI',   Icon: BrainCircuit    },
  { id: 'analytics',  label: 'Analytics',    Icon: BarChart2       },
]

export default function Layout({ activePage, setActivePage, voiceState, onVoiceToggle, children }) {
  const VIcon = voiceState.enabled ? Mic : MicOff

  return (
    <div style={{ display: 'flex', height: '100%', background: 'var(--bg-0)', overflow: 'hidden' }}>

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside
        style={{
          width: 220, flexShrink: 0, height: '100%',
          display: 'flex', flexDirection: 'column',
          background: 'var(--bg-1)',
          borderRight: '1px solid var(--cyan-border)',
        }}
      >
        {/* Logo */}
        <div style={{
          padding: '20px 18px 16px',
          borderBottom: '1px solid var(--cyan-border)',
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'var(--cyan-dim)',
            border: '1px solid var(--cyan-border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            <Eye size={18} color="var(--cyan)" strokeWidth={1.8} />
          </div>
          <div>
            <p style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-1)', lineHeight: 1.1 }}>
              ZYGLASS
            </p>
            <p style={{ fontSize: 9, fontWeight: 500, letterSpacing: '0.2em', color: 'var(--cyan)', opacity: 0.7 }}>
              VISION OS
            </p>
          </div>
        </div>

        {/* Nav items */}
        <nav style={{ flex: 1, padding: '10px 10px', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {NAV.map(({ id, label, Icon }) => {
            const active = activePage === id
            return (
              <motion.button
                key={id}
                onClick={() => setActivePage(id)}
                style={{
                  position: 'relative',
                  display:  'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '9px 10px',
                  borderRadius: 8,
                  border: 'none',
                  background: active ? 'rgba(34,211,238,0.10)' : 'transparent',
                  color: active ? 'var(--cyan)' : 'var(--text-2)',
                  cursor: 'pointer',
                  transition: 'background 0.2s, color 0.2s',
                  overflow: 'hidden',
                  textAlign: 'left',
                  width: '100%',
                }}
                whileHover={{ x: 2, backgroundColor: active ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.04)' }}
                whileTap={{ scale: 0.97 }}
              >
                {active && (
                  <motion.span
                    layoutId="sidebar-pill"
                    style={{
                      position: 'absolute', left: 0, top: '10%', bottom: '10%',
                      width: 3, borderRadius: '0 2px 2px 0',
                      background: 'var(--cyan)',
                    }}
                    transition={{ type: 'spring', stiffness: 400, damping: 35 }}
                  />
                )}
                <Icon size={16} strokeWidth={active ? 2 : 1.6} />
                <span style={{ fontSize: 13, fontWeight: active ? 600 : 400, letterSpacing: '0.02em' }}>
                  {label}
                </span>
                {active && <ChevronRight size={12} style={{ marginLeft: 'auto', opacity: 0.5 }} />}
              </motion.button>
            )
          })}
        </nav>

        {/* Footer: voice + status */}
        <div style={{
          padding: '12px 10px',
          borderTop: '1px solid var(--cyan-border)',
          display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          {/* Voice toggle */}
          <motion.button
            onClick={onVoiceToggle}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 10px', borderRadius: 8, border: 'none',
              background: voiceState.enabled
                ? (voiceState.wake_detected
                    ? 'rgba(251,191,36,0.12)'
                    : voiceState.listening
                      ? 'rgba(74,222,128,0.10)'
                      : 'rgba(34,211,238,0.08)')
                : 'rgba(255,255,255,0.04)',
              cursor: 'pointer', width: '100%', textAlign: 'left',
              transition: 'background 0.3s',
            }}
            whileTap={{ scale: 0.97 }}
          >
            <VIcon
              size={14}
              color={
                voiceState.wake_detected ? 'var(--amber)'
                  : voiceState.listening  ? 'var(--green)'
                  : voiceState.enabled    ? 'var(--cyan)'
                  : 'var(--text-3)'
              }
              style={voiceState.listening ? { animation: 'glow-breathe 1.5s ease-in-out infinite' } : {}}
            />
            <span style={{
              fontSize: 11, fontWeight: 500, letterSpacing: '0.06em',
              color: voiceState.enabled ? 'var(--text-1)' : 'var(--text-3)',
              flex: 1,
            }}>
              {voiceState.wake_detected
                ? 'WAKE WORD!'
                : voiceState.listening
                  ? 'Listening…'
                  : voiceState.enabled
                    ? 'Voice On'
                    : 'Voice Off'}
            </span>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: voiceState.enabled ? 'var(--green)' : 'var(--text-3)',
              boxShadow: voiceState.listening ? '0 0 8px var(--green)' : 'none',
            }} />
          </motion.button>

          {/* Online indicator */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '4px 10px',
            fontSize: 10, color: 'var(--text-3)',
          }}>
            <Signal size={11} color="var(--green)" />
            <span style={{ letterSpacing: '0.08em' }}>BACKEND ONLINE</span>
          </div>
        </div>
      </aside>

      {/* ── Main content ────────────────────────────────────────────── */}
      <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {children}
      </main>
    </div>
  )
}
