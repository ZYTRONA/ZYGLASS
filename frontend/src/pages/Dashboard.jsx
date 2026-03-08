/**
 * Dashboard — primary control surface.
 * Layout: video (center) · data console (right) · mode rail (bottom).
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ScanSearch, BookOpen, Smile, HandMetal, Users,
  DollarSign, NotebookPen, Play, Square,
  Activity, ChevronRight, UserPlus, Trash2,
} from 'lucide-react'
import VideoFeed from '../components/VideoFeed'

const MODE_META = [
  {
    id: 1, key: 'insight',
    label: 'Insight Explorer',
    short: 'Object Detection',
    Icon: ScanSearch,
    color: 'var(--cyan)',
    dim:   'rgba(34,211,238,0.10)',
    border:'rgba(34,211,238,0.25)',
  },
  {
    id: 2, key: 'reader',
    label: 'Multi Reader',
    short: 'OCR + Translate',
    Icon: BookOpen,
    color: 'var(--purple)',
    dim:   'rgba(168,85,247,0.10)',
    border:'rgba(168,85,247,0.25)',
  },
  {
    id: 3, key: 'emotion',
    label: 'Emotion AI',
    short: 'Facial Analysis',
    Icon: Smile,
    color: '#fb923c',
    dim:   'rgba(251,146,60,0.10)',
    border:'rgba(251,146,60,0.25)',
  },
  {
    id: 4, key: 'sign',
    label: 'Sign Translator',
    short: 'Medical Signs',
    Icon: HandMetal,
    color: 'var(--green)',
    dim:   'rgba(74,222,128,0.10)',
    border:'rgba(74,222,128,0.25)',
  },
]

const S = {
  root: {
    display: 'flex', flexDirection: 'column', height: '100%',
    background: 'var(--bg-0)', overflow: 'hidden',
  },
  topBar: {
    padding: '10px 20px',
    borderBottom: '1px solid var(--cyan-border)',
    display: 'flex', alignItems: 'center', gap: 12,
    background: 'var(--bg-1)', flexShrink: 0,
  },
  body: {
    flex: 1, display: 'grid', overflow: 'hidden',
    gridTemplateColumns: '1fr 300px',
    gridTemplateRows:    '1fr 72px',
    gap: 0,
  },
  videoWrap: {
    gridColumn: '1', gridRow: '1',
    padding: '14px 14px 7px 14px', overflow: 'hidden',
    display: 'flex', flexDirection: 'column', gap: 8,
    alignItems: 'center', justifyContent: 'center',
  },
  videoContainer: {
    width: '100%', maxWidth: '75%',
    display: 'flex', flexDirection: 'column',
    flex: 1,
  },
  console: {
    gridColumn: '2', gridRow: '1 / 3',
    borderLeft: '1px solid var(--cyan-border)',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    background: 'var(--bg-1)',
  },
  modeRail: {
    gridColumn: '1', gridRow: '2',
    borderTop: '1px solid var(--cyan-border)',
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '0 14px', overflow: 'hidden', flexShrink: 0,
    background: 'var(--bg-1)',
  },
}

function ModeBtn({ meta, active, onClick }) {
  const { label, Icon, color, dim, border } = meta
  return (
    <motion.button
      onClick={onClick}
      whileTap={{ scale: 0.95 }}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
        padding: '6px 10px', borderRadius: 10, border: `1px solid ${active ? border : 'transparent'}`,
        background: active ? dim : 'transparent', cursor: 'pointer',
        flexShrink: 0, transition: 'all 0.2s', minWidth: 80,
      }}
    >
      <Icon size={16} color={active ? color : 'var(--text-3)'} strokeWidth={active ? 2 : 1.5} />
      <span style={{
        fontSize: 9, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
        color: active ? color : 'var(--text-3)', whiteSpace: 'nowrap',
      }}>
        {label}
      </span>
    </motion.button>
  )
}

function Tag({ children, color = 'var(--cyan)' }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 7px', borderRadius: 4,
      background: `color-mix(in srgb, ${color} 14%, transparent)`,
      border: `1px solid color-mix(in srgb, ${color} 30%, transparent)`,
      fontSize: 10, fontWeight: 600, color, letterSpacing: '0.06em',
    }}>
      {children}
    </span>
  )
}

export default function Dashboard({
  activeMode, setActiveMode, streamActive, setStreamActive,
  ocrData, signData, faceData,
  currencyMode, lectureMode, currencyData, lectureData,
  toggleCurrency,
  enrollName, setEnrollName, enrollUnknown, enrollStatus, enrollMsg,
  voiceState,
}) {
  const [fullscreen, setFullscreen] = useState(false)
  const meta = MODE_META.find(m => m.id === activeMode) ?? MODE_META[0]

  // ── Fullscreen overlay ──────────────────────────────────────────────────────
  if (fullscreen) return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, background: '#000' }}>
      <VideoFeed active={streamActive} className="absolute inset-0 w-full h-full rounded-none" />
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)',
        padding: '24px 20px 16px', display: 'flex', gap: 8, alignItems: 'center',
      }}>
        {MODE_META.map(m => (
          <ModeBtn key={m.id} meta={m} active={activeMode === m.id} onClick={() => setActiveMode(m.id)} />
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            onClick={() => setStreamActive(v => !v)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '7px 14px', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(0,0,0,0.4)',
              color: 'var(--text-1)', cursor: 'pointer', fontSize: 12,
            }}
          >
            {streamActive ? <Square size={12} /> : <Play size={12} />}
            {streamActive ? 'Stop' : 'Start'}
          </button>
          <button
            onClick={() => setFullscreen(false)}
            style={{
              padding: '7px 14px', borderRadius: 8,
              border: '1px solid rgba(34,211,238,0.4)', background: 'rgba(34,211,238,0.1)',
              color: 'var(--cyan)', cursor: 'pointer', fontSize: 12,
            }}
          >
            Exit ⎋
          </button>
        </div>
      </div>
    </div>
  )

  return (
    <div style={S.root}>

      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <div style={S.topBar}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: meta.color, boxShadow: `0 0 8px ${meta.color}`,
          flexShrink: 0,
        }} />
        <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-2)', textTransform: 'uppercase' }}>
          Dashboard
        </p>
        <ChevronRight size={12} color="var(--text-3)" />
        <p style={{ fontSize: 12, color: meta.color, fontWeight: 600 }}>
          {meta.label}
        </p>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Stream toggle */}
          <motion.button
            onClick={() => setStreamActive(v => !v)}
            whileTap={{ scale: 0.95 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 7,
              border: `1px solid ${streamActive ? 'rgba(248,113,113,0.35)' : 'rgba(34,211,238,0.35)'}`,
              background: streamActive ? 'rgba(248,113,113,0.08)' : 'rgba(34,211,238,0.08)',
              color: streamActive ? 'var(--red)' : 'var(--cyan)',
              cursor: 'pointer', fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
            }}
          >
            {streamActive ? <Square size={11} /> : <Play size={11} />}
            {streamActive ? 'Stop' : 'Start'}
          </motion.button>

          {/* Activity indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: 'var(--text-3)' }}>
            <Activity size={11} color="var(--green)" />
            <span style={{ letterSpacing: '0.08em' }}>LIVE</span>
          </div>
        </div>
      </div>

      {/* ── Body grid ───────────────────────────────────────────────────── */}
      <div style={S.body}>

        {/* Video */}
        <div style={S.videoWrap}>
          <div style={S.videoContainer}>
            <VideoFeed
              active={streamActive}
              onFullscreen={() => setFullscreen(true)}
              className="flex-1"
            />
          </div>
        </div>

        {/* Right console */}
        <div style={S.console}>
          <div style={{
            padding: '12px 14px 8px', borderBottom: '1px solid var(--cyan-border)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-2)', textTransform: 'uppercase' }}>
              Console
            </span>
            <Tag>{meta.short}</Tag>
          </div>

          {/* Sub-mode toggles */}
          {activeMode === 1 && (
            <div style={{ padding: '10px 14px 6px', borderBottom: '1px solid rgba(34,211,238,0.06)' }}>
              <motion.button
                onClick={toggleCurrency}
                whileTap={{ scale: 0.97 }}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px',
                  borderRadius: 8, border: `1px solid ${currencyMode ? 'rgba(251,191,36,0.4)' : 'rgba(255,255,255,0.07)'}`,
                  background: currencyMode ? 'rgba(251,191,36,0.08)' : 'transparent',
                  color: currencyMode ? 'var(--amber)' : 'var(--text-3)', cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                <DollarSign size={13} />
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.06em' }}>
                  Currency Reader
                </span>
                <div style={{
                  marginLeft: 'auto', fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
                  color: currencyMode ? 'var(--green)' : 'var(--text-3)',
                }}>
                  {currencyMode ? '● ON' : '○ OFF'}
                </div>
              </motion.button>
            </div>
          )}

          {/* Data panel */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            <AnimatePresence mode="wait">
              <motion.div
                key={activeMode}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >

                {/* Mode 1 */}
                {activeMode === 1 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Label>Object Detection</Label>
                    <Muted>YOLOv10n · NMS-free · GPU accelerated</Muted>
                    {currencyMode && (
                      <div style={{ marginTop: 4 }}>
                        <Label color="var(--amber)">Currency Reader</Label>
                        {currencyData.last_denomination !== 'None' && currencyData.last_denomination ? (
                          <div style={{ marginTop: 6, padding: '8px 10px', borderRadius: 8, background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)' }}>
                            <p style={{ fontSize: 22, fontWeight: 700, color: 'var(--amber)' }}>
                              ₹{currencyData.last_denomination}
                            </p>
                          </div>
                        ) : (
                          <Muted>Hold banknote in frame…</Muted>
                        )}
                        {currencyData.detections?.slice(-4).map((d, i) => (
                          <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-2)', padding: '2px 0' }}>
                            <span>₹{d.denomination}</span>
                            <span style={{ color: 'var(--amber)' }}>{Math.round(d.confidence * 100)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Mode 2 */}
                {activeMode === 2 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Label>OCR / Translation</Label>
                    {ocrData.length === 0 ? (
                      <Muted>Hold printed text in frame…</Muted>
                    ) : ocrData.map((item, i) => (
                      <div key={i} style={{
                        padding: '7px 10px', borderRadius: 8, borderLeft: '2px solid var(--cyan)',
                        background: 'rgba(34,211,238,0.04)',
                      }}>
                        <p style={{ fontSize: 12, color: 'var(--text-1)', fontWeight: 500 }}>{item.text}</p>
                        {item.translation && <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>→ {item.translation}</p>}
                        <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 2 }}>{item.confidence}% conf</p>
                      </div>
                    ))}
                    {lectureMode && (
                      <>
                        <Label color="var(--purple)">Lecture Notes · {lectureData.count} lines</Label>
                        {lectureData.lines.slice(-6).reverse().map((line, i) => (
                          <p key={i} style={{ fontSize: 11, color: 'var(--text-2)', borderLeft: '2px solid rgba(168,85,247,0.5)', paddingLeft: 8 }}>{line}</p>
                        ))}
                        {lectureData.summary && (
                          <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(168,85,247,0.06)', border: '1px solid rgba(168,85,247,0.2)' }}>
                            <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--purple)', marginBottom: 4 }}>AI SUMMARY</p>
                            <p style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>{lectureData.summary}</p>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}

                {/* Mode 3 */}
                {activeMode === 3 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Label>Emotion Analyzer</Label>
                    <Muted>ResNet50 · 7 emotion classes</Muted>
                    <Muted>Look into camera for analysis</Muted>
                  </div>
                )}

                {/* Mode 4 */}
                {activeMode === 4 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Label>Sign Translator</Label>
                    {signData.length === 0
                      ? <Muted>Show a hand sign to the camera…</Muted>
                      : <>
                          <div style={{ padding: '10px', borderRadius: 8, background: 'rgba(74,222,128,0.06)', border: '1px solid rgba(74,222,128,0.2)' }}>
                            <p style={{ fontSize: 9, letterSpacing: '0.1em', color: 'var(--green)', fontWeight: 700 }}>LAST DETECTED</p>
                            <p style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-1)', marginTop: 2 }}>
                              {signData[signData.length - 1]?.sign}
                            </p>
                          </div>
                          {[...signData].reverse().slice(0, 6).map((e, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: e.sign === 'No Hand Detected' ? 'var(--text-3)' : 'var(--text-2)' }}>
                              <span>{e.sign}</span>
                              <span style={{ color: 'var(--text-3)' }}>{e.time}</span>
                            </div>
                          ))}
                        </>
                    }
                    <Muted>Random Forest · 63 landmark features</Muted>
                  </div>
                )}

                {/* Mode 5 */}
                {activeMode === 5 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Label>Social Memory</Label>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)', flex: 1 }}>
                        <p style={{ fontSize: 9, letterSpacing: '0.1em', color: 'var(--amber)', fontWeight: 700 }}>ENROLLED</p>
                        <p style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-1)' }}>{faceData.known_count}</p>
                      </div>
                      <div style={{ padding: '8px 10px', borderRadius: 8, background: 'rgba(34,211,238,0.06)', border: '1px solid rgba(34,211,238,0.2)', flex: 1 }}>
                        <p style={{ fontSize: 9, letterSpacing: '0.1em', color: 'var(--cyan)', fontWeight: 700 }}>LAST SEEN</p>
                        <p style={{ fontSize: 12, fontWeight: 600, color: faceData.last_face === 'Unknown' ? 'var(--red)' : 'var(--text-1)', marginTop: 2 }}>{faceData.last_face}</p>
                      </div>
                    </div>
                    {faceData.last_face === 'Unknown' && (
                      <div style={{ padding: '10px', borderRadius: 8, background: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.3)' }}>
                        <p style={{ fontSize: 10, color: 'var(--red)', fontWeight: 700, marginBottom: 6 }}>Unknown person detected</p>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <input
                            value={enrollName} onChange={e => setEnrollName(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && enrollUnknown()}
                            placeholder="Enter name…"
                            style={{
                              flex: 1, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                              borderRadius: 6, padding: '5px 8px', color: 'var(--text-1)', fontSize: 12, outline: 'none',
                              fontFamily: 'inherit',
                            }}
                          />
                          <motion.button
                            onClick={enrollUnknown}
                            whileTap={{ scale: 0.95 }}
                            disabled={enrollStatus === 'loading'}
                            style={{
                              padding: '5px 10px', borderRadius: 6, border: 'none',
                              background: 'var(--cyan)', color: '#000', fontSize: 12, fontWeight: 700, cursor: 'pointer',
                            }}
                          >
                            <UserPlus size={12} />
                          </motion.button>
                        </div>
                        {enrollMsg && (
                          <p style={{ fontSize: 10, marginTop: 4, color: enrollStatus === 'ok' ? 'var(--green)' : 'var(--red)' }}>
                            {enrollMsg}
                          </p>
                        )}
                      </div>
                    )}
                    {faceData.history.slice(-5).reverse().map((e, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-2)' }}>
                        <span style={{ color: e.name === 'Unknown' ? 'var(--red)' : 'var(--green)' }}>{e.name}</span>
                        <span style={{ color: 'var(--text-3)' }}>{e.time}</span>
                      </div>
                    ))}
                  </div>
                )}

              </motion.div>
            </AnimatePresence>
          </div>

          {/* Voice reply strip */}
          {voiceState.enabled && voiceState.last_response && (
            <div style={{
              padding: '10px 14px', borderTop: '1px solid var(--cyan-border)',
              background: 'rgba(34,211,238,0.04)',
            }}>
              <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--cyan)', marginBottom: 4 }}>
                ZYGLASS AI REPLY
              </p>
              <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>{voiceState.last_response}</p>
            </div>
          )}
        </div>

        {/* ── Mode rail ────────────────────────────────────────────────── */}
        <div style={S.modeRail}>
          {MODE_META.map(m => (
            <ModeBtn key={m.id} meta={m} active={activeMode === m.id} onClick={() => setActiveMode(m.id)} />
          ))}
        </div>

      </div>
    </div>
  )
}

// ── Tiny helpers ────────────────────────────────────────────────────────────
function Label({ children, color = 'var(--cyan)' }) {
  return (
    <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color, textTransform: 'uppercase' }}>
      {children}
    </p>
  )
}
function Muted({ children }) {
  return <p style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.6 }}>{children}</p>
}
