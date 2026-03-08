/**
 * ModesPage — detailed mode cards + sub-mode controls.
 * Each card: icon, name, description, features list, live stats, activate button.
 */
import { motion, AnimatePresence } from 'framer-motion'
import {
  ScanSearch, BookOpen, Smile, HandMetal, Users,
  DollarSign, NotebookPen, Zap, CheckCircle2,
  ArrowRight, ChevronDown, Info, Camera,
} from 'lucide-react'
import { useState } from 'react'

const MODES = [
  {
    id: 1,
    label:       'Insight Explorer',
    tagline:     'See the world through AI eyes',
    description: 'Real-time object detection powered by YOLOv10n — NMS-free architecture for ultra-low-latency inference. Runs on GPU via ONNX runtime.',
    Icon:        ScanSearch,
    color:       'var(--cyan)',
    dim:         'rgba(34,211,238,0.08)',
    border:      'rgba(34,211,238,0.20)',
    features: [
      'YOLOv10n ONNX model (NMS-free, <2 ms inference)',
      'GPU-accelerated via CUDA / ONNX runtime',
      'Voice announcements for detected objects',
      '80 COCO classes including people, vehicles, items',
    ],
    subModes: [
      {
        key:   'currency',
        label: 'Currency Reader',
        desc:  'Detects Indian Rupee banknotes and announces denomination via TTS. Requires rupee_model.pt.',
        Icon:  DollarSign,
        color: 'var(--amber)',
      },
    ],
  },
  {
    id: 2,
    label:       'Multilingual Reader',
    tagline:     'OCR with AI translation',
    description: 'Uses RapidOCR (ONNX PaddleOCR) for text extraction, Google Translate for real-time translation, and an optional Ollama LLM for medical simplification.',
    Icon:        BookOpen,
    color:       'var(--purple)',
    dim:         'rgba(168,85,247,0.08)',
    border:      'rgba(168,85,247,0.20)',
    features: [
      'RapidOCR ONNX — no PaddleOCR/Python overhead',
      'Google Translate for detected text',
      'Stability filter — requires 2 consecutive detections',
      'Ollama LLM for medical text simplification',
    ],
    subModes: [],
  },
  {
    id: 3,
    label:       'Emotion Analyzer',
    tagline:     'Understand the emotional state of anyone',
    description: 'ResNet50 trained on a 7-class emotion dataset. Compiled to TorchScript for ~25% faster inference. Anti-flicker temporal smoothing over a 5-frame window.',
    Icon:        Smile,
    color:       '#fb923c',
    dim:         'rgba(251,146,60,0.08)',
    border:      'rgba(251,146,60,0.20)',
    features: [
      'ResNet50 — 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise',
      'TorchScript compilation for faster inference',
      '5-frame temporal smoothing to prevent flicker',
      'Voice announcement on emotion change',
    ],
    subModes: [],
  },
  {
    id: 4,
    label:       'Medical Sign Translator',
    tagline:     'Hand signs to spoken medical vocabulary',
    description: 'MediaPipe HandLandmarker extracts 21 3D joint positions (63 features). A Random Forest classifier maps them to medical signs and speaks via pyttsx3.',
    Icon:        HandMetal,
    color:       'var(--green)',
    dim:         'rgba(74,222,128,0.08)',
    border:      'rgba(74,222,128,0.20)',
    features: [
      'MediaPipe HandLandmarker — 21 joints × 3 axes',
      'Random Forest classifier (sklearn joblib)',
      'Majority vote over 9-frame window for stability',
      'pyttsx3 offline TTS — works without internet',
    ],
    subModes: [],
  },
]

function FeatureRow({ text }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', padding: '3px 0' }}>
      <CheckCircle2 size={12} style={{ color: 'var(--green)', flexShrink: 0, marginTop: 2 }} />
      <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>{text}</p>
    </div>
  )
}

function ModeCard({ mode, isActive, onActivate }) {
  const [expanded, setExpanded] = useState(false)
  const [captureStatus, setCaptureStatus] = useState('')
  const { Icon, color, dim, border } = mode

  const captureTrainingSample = async () => {
    try {
      setCaptureStatus('Capturing...')
      const response = await fetch('/save_training_frame')
      const result = await response.json()
      
      if (result.status === 'success') {
        setCaptureStatus(`✓ Saved ${result.mode}`)
        setTimeout(() => setCaptureStatus(''), 2000)
      } else {
        setCaptureStatus('✗ Failed')
        setTimeout(() => setCaptureStatus(''), 2000)
      }
    } catch (error) {
      console.error('Training capture error:', error)
      setCaptureStatus('✗ Error')
      setTimeout(() => setCaptureStatus(''), 2000)
    }
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        borderRadius: 14, border: `1px solid ${isActive ? border : 'rgba(255,255,255,0.06)'}`,
        background: isActive ? dim : 'var(--bg-2)',
        overflow: 'hidden',
        transition: 'border-color 0.3s, background 0.3s',
        boxShadow: isActive ? `0 0 20px color-mix(in srgb, ${color} 15%, transparent)` : 'none',
      }}
    >
      {/* Card header */}
      <div style={{ padding: '16px 18px', display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        <div style={{
          width: 42, height: 42, borderRadius: 12, flexShrink: 0,
          background: isActive ? `color-mix(in srgb, ${color} 20%, transparent)` : 'rgba(255,255,255,0.05)',
          border: `1px solid ${isActive ? border : 'rgba(255,255,255,0.07)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: isActive ? `0 0 12px color-mix(in srgb, ${color} 30%, transparent)` : 'none',
          transition: 'all 0.3s',
        }}>
          <Icon size={20} color={isActive ? color : 'var(--text-3)'} strokeWidth={isActive ? 2 : 1.5} />
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
            <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-3)', textTransform: 'uppercase' }}>
              Mode {mode.id}
            </span>
            {isActive && (
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                style={{
                  padding: '1px 7px', borderRadius: 4, fontSize: 9, fontWeight: 700,
                  letterSpacing: '0.1em', background: color, color: '#000',
                }}
              >
                ACTIVE
              </motion.span>
            )}
          </div>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-1)', marginBottom: 2 }}>{mode.label}</h3>
          <p style={{ fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>{mode.tagline}</p>
        </div>

        {/* Activate and Training buttons */}
        <div style={{ display: 'flex', gap: 8 }}>
          <motion.button
            onClick={onActivate}
            whileTap={{ scale: 0.93 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 5, padding: '7px 14px',
              borderRadius: 8, border: `1px solid ${isActive ? color : 'rgba(255,255,255,0.12)'}`,
              background: isActive ? `color-mix(in srgb, ${color} 15%, transparent)` : 'rgba(255,255,255,0.04)',
              color: isActive ? color : 'var(--text-2)',
              cursor: isActive ? 'default' : 'pointer',
              fontSize: 11, fontWeight: 600, letterSpacing: '0.06em', flexShrink: 0,
              transition: 'all 0.25s', flex: 1,
            }}
            disabled={isActive}
          >
            {isActive ? <Zap size={12} /> : <ArrowRight size={12} />}
            {isActive ? 'Active' : 'Activate'}
          </motion.button>

          <motion.button
            onClick={captureTrainingSample}
            whileTap={{ scale: 0.93 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 5, padding: '7px 12px',
              borderRadius: 8, border: '1px solid rgba(34,197,94,0.3)',
              background: 'rgba(34,197,94,0.08)',
              color: 'var(--green)',
              cursor: 'pointer',
              fontSize: 10, fontWeight: 600, letterSpacing: '0.06em',
              transition: 'all 0.25s', minWidth: 80, justifyContent: 'center',
            }}
            title="Capture current frame for training data"
          >
            <Camera size={11} />
            {captureStatus || 'Train'}
          </motion.button>
        </div>
      </div>

      {/* Description + expand */}
      <div style={{ padding: '0 18px 14px' }}>
        <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.65, marginBottom: 10 }}>
          {mode.description}
        </p>

        {/* Expand features */}
        <button
          onClick={() => setExpanded(v => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-3)',
            fontSize: 11, cursor: 'pointer', background: 'none', border: 'none',
            padding: 0, fontFamily: 'inherit',
          }}
        >
          <Info size={12} />
          <span>{expanded ? 'Hide' : 'Show'} features</span>
          <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
            <ChevronDown size={12} />
          </motion.span>
        </button>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              style={{ overflow: 'hidden' }}
            >
              <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 2 }}>
                {mode.features.map((f, i) => <FeatureRow key={i} text={f} />)}
              </div>

              {/* Sub-modes */}
              {mode.subModes.map(sub => (
                <div key={sub.key} style={{
                  marginTop: 10, padding: '10px 12px', borderRadius: 10,
                  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
                }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                    <sub.Icon size={13} color={sub.color} />
                    <p style={{ fontSize: 12, fontWeight: 600, color: sub.color }}>{sub.label}</p>
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.55 }}>{sub.desc}</p>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default function ModesPage({ activeMode, setActiveMode }) {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-0)' }}>

      {/* Top bar */}
      <div style={{
        padding: '14px 24px', borderBottom: '1px solid var(--cyan-border)',
        background: 'var(--bg-1)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-2)', textTransform: 'uppercase' }}>
          Vision Modes
        </p>
        <span style={{
          padding: '2px 8px', borderRadius: 6, fontSize: 10, fontWeight: 700,
          background: 'rgba(34,211,238,0.12)', border: '1px solid rgba(34,211,238,0.25)',
          color: 'var(--cyan)',
        }}>
          {MODES.length} modes
        </span>
        <p style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-3)' }}>
          Active: <span style={{ color: 'var(--text-1)', fontWeight: 600 }}>
            {MODES.find(m => m.id === activeMode)?.label}
          </span>
        </p>
      </div>

      {/* Card grid */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: 20,
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
        gap: 14,
        alignContent: 'start',
      }}>
        {MODES.map((mode, idx) => (
          <motion.div
            key={mode.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.06, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          >
            <ModeCard
              mode={mode}
              isActive={activeMode === mode.id}
              onActivate={() => setActiveMode(mode.id)}
            />
          </motion.div>
        ))}
      </div>

    </div>
  )
}
