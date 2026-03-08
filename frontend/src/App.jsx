/**
 * App — root of the ZYGLASS VISION OS SPA.
 *
 * Global state is held here and passed as props to pages.
 * Polling is consolidated into a single setInterval per data stream.
 * Page routing is pure React state — no router dependency.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Layout      from './components/Layout'
import Dashboard   from './pages/Dashboard'
import ModesPage   from './pages/ModesPage'
import AIAssistant from './pages/AIAssistant'
import Analytics   from './pages/Analytics'
import { api }     from './api'

// Page transition variants
const PAGE_VARIANTS = {
  initial: { opacity: 0, x: 18 },
  animate: { opacity: 1, x: 0,  transition: { duration: 0.28, ease: [0.22, 1, 0.36, 1] } },
  exit:    { opacity: 0, x: -14, transition: { duration: 0.18 } },
}

export default function App() {
  // ── Navigation ─────────────────────────────────────────────────────────────
  const [activePage, setActivePage] = useState('dashboard')

  // ── Mode ────────────────────────────────────────────────────────────────────
  const [activeMode, _setActiveMode] = useState(1)

  const setActiveMode = useCallback(async (id) => {
    _setActiveMode(id)
    try { await api.setMode(id) } catch { /* network error, local state already updated */ }
  }, [])

  // ── Stream ──────────────────────────────────────────────────────────────────
  const [streamActive, setStreamActive] = useState(true)

  // ── Voice ───────────────────────────────────────────────────────────────────
  const [voiceState, setVoiceState] = useState({
    enabled: false, listening: false, wake_detected: false,
    last_command: '', last_response: '', last_ts: '', active: false,
  })
  const voiceRef = useRef(voiceState)

  const toggleVoice = useCallback(async () => {
    console.log('[VOICE] Toggling voice, current state:', voiceRef.current.enabled)
    try {
      const d = await api.toggleVoice(!voiceRef.current.enabled)
      console.log('[VOICE] Backend response:', d)
      setVoiceState(prev => ({ ...prev, enabled: d.voice_enabled ?? !prev.enabled }))
    } catch (err) {
      console.error('[VOICE] Toggle failed:', err)
      setVoiceState(prev => ({ ...prev, enabled: !prev.enabled }))
    }
  }, [])

  // ── Sub-modes ────────────────────────────────────────────────────────────────
  const [currencyMode, setCurrencyMode] = useState(false)
  const [lectureMode,  setLectureMode]  = useState(false)

  // ── Mode-specific data (declared before sub-mode callbacks that reference setLectureData)
  const [ocrData,      setOcrData]      = useState([])
  const [signData,     setSignData]     = useState([])
  const [faceData,     setFaceData]     = useState({ history: [], last_face: 'None', known_count: 0, known_names: [] })
  const [currencyData, setCurrencyData] = useState({ detections: [], last_denomination: 'None', notes_detected: 0, model_available: false })
  const [lectureData,  setLectureData]  = useState({ lines: [], summary: '', count: 0 })

  const toggleCurrency = useCallback(async () => {
    try {
      const d = await api.toggleCurrency(!currencyMode)
      setCurrencyMode(d.currency_mode ?? !currencyMode)
    } catch { setCurrencyMode(v => !v) }
  }, [currencyMode])

  const toggleLecture = useCallback(async (enable, clear = false) => {
    try {
      const d = await api.toggleLecture(enable, clear)
      setLectureMode(d.lecture_mode ?? enable)
      if (clear) setLectureData({ lines: [], summary: '', count: 0 })
    } catch { setLectureMode(v => !v) }
  }, [setLectureData])

  // ── Enroll ───────────────────────────────────────────────────────────────────
  const [enrollName,   setEnrollName]   = useState('')
  const [enrollStatus, setEnrollStatus] = useState('')
  const [enrollMsg,    setEnrollMsg]    = useState('')

  const enrollUnknown = useCallback(async () => {
    if (!enrollName.trim()) return
    setEnrollStatus('loading')
    try {
      const d = await api.enrollUnknown(enrollName.trim())
      if (d.status === 'success') {
        setEnrollStatus('ok')
        setEnrollMsg(d.message)
        setEnrollName('')
      } else {
        setEnrollStatus('err')
        setEnrollMsg(d.message)
      }
    } catch {
      setEnrollStatus('err')
      setEnrollMsg('Backend unreachable')
    }
    setTimeout(() => { setEnrollStatus(''); setEnrollMsg('') }, 4000)
  }, [enrollName])

  // ── Polling: mode-specific data (1 s) ───────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        if (activeMode === 2) {
          const d = await api.ocrData()
          setOcrData(d.results ?? [])
        } else if (activeMode === 4) {
          const d = await api.signData()
          setSignData(d.history ?? [])
        } else if (activeMode === 5) {
          const d = await api.faceData()
          setFaceData({
            history:     d.history     ?? [],
            last_face:   d.last_face   ?? 'None',
            known_count: d.known_count ?? 0,
            known_names: d.known_names ?? [],
          })
        }
        if (activeMode === 1 && currencyMode) {
          const d = await api.currencyData()
          setCurrencyData({
            detections:       d.detections       ?? [],
            last_denomination: d.last_denomination ?? 'None',
            notes_detected:   d.notes_detected   ?? 0,
            model_available:  d.model_available  ?? false,
          })
        }
        if (activeMode === 2 && lectureMode) {
          const d = await api.lectureData()
          setLectureData({ lines: d.lines ?? [], summary: d.summary ?? '', count: d.count ?? 0 })
        }
      } catch { /* backend unreachable, keep stale data */ }
    }

    poll()
    const iv = setInterval(poll, 1000)
    return () => clearInterval(iv)
  }, [activeMode, currencyMode, lectureMode])

  // ── Polling: voice state (500 ms, diff-only re-render) ───────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const d = await api.voiceData()
        const p = voiceRef.current
        if (
          d.enabled       !== p.enabled       ||
          d.listening     !== p.listening     ||
          d.wake_detected !== p.wake_detected ||
          d.last_command  !== p.last_command  ||
          d.last_response !== p.last_response
        ) {
          console.log('[VOICE] State update:', {
            enabled: d.enabled,
            listening: d.listening,
            wake_detected: d.wake_detected,
            last_command: d.last_command
          })
          voiceRef.current = d
          setVoiceState(d)
        }
      } catch { /* backend unreachable */ }
    }

    const iv = setInterval(poll, 500)
    return () => clearInterval(iv)
  }, [])

  // ── Page render ──────────────────────────────────────────────────────────────
  const sharedDashProps = {
    activeMode, setActiveMode, streamActive, setStreamActive,
    ocrData, signData, faceData,
    currencyMode, lectureMode, currencyData, lectureData,
    toggleCurrency, toggleLecture,
    enrollName, setEnrollName, enrollUnknown, enrollStatus, enrollMsg,
    voiceState,
  }

  return (
    <Layout
      activePage={activePage}
      setActivePage={setActivePage}
      voiceState={voiceState}
      onVoiceToggle={toggleVoice}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={activePage}
          variants={PAGE_VARIANTS}
          initial="initial"
          animate="animate"
          exit="exit"
          style={{ height: '100%', overflow: 'hidden' }}
        >
          {activePage === 'dashboard'  && <Dashboard  {...sharedDashProps} />}
          {activePage === 'modes'      && <ModesPage  activeMode={activeMode} setActiveMode={setActiveMode} />}
          {activePage === 'ai'         && <AIAssistant voiceState={voiceState} onVoiceToggle={toggleVoice} />}
          {activePage === 'analytics'  && <Analytics />}
        </motion.div>
      </AnimatePresence>
    </Layout>
  )
}

