/**
 * AIAssistant — Complete Voice-Powered AI Interface for ZyGlass
 * 
 * Features:
 *  • Real-time voice visualization with 3D orb
 *  • Conversational chat interface
 *  • Smart command suggestions based on current mode
 *  • Voice activity waveform
 *  • Context-aware responses
 *  • Mode switching via voice
 *  • Live transcription display
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Mic, MicOff, Radio, Zap, MessageSquare, Send, Sparkles,
  Volume2, VolumeX, CornerDownRight, Clock, Activity,
  Eye, FileText, Smile, Hand, Users, Settings, Play,
  Pause, RotateCcw, ChevronDown, Info, Loader2
} from 'lucide-react'

// Mode icons mapping
const MODE_ICONS = {
  1: Eye,      // Object Detection
  2: FileText, // OCR Reader
  3: Smile,    // Emotion Analysis
  4: Hand,     // Sign Language
  5: Users,    // Face Recognition
}

// Smart command suggestions based on context
const COMMAND_TEMPLATES = {
  general: [
    { text: 'What mode am I in?', icon: Info },
    { text: 'Switch to reading mode', icon: FileText },
    { text: 'What do you see?', icon: Eye },
  ],
  1: [ // Object Detection
    { text: 'What objects do you see?', icon: Eye },
    { text: 'Count the items', icon: Activity },
    { text: 'Describe the scene', icon: MessageSquare },
  ],
  2: [ // OCR
    { text: 'Read the text', icon: FileText },
    { text: 'Translate to English', icon: FileText },
    { text: 'Start lecture mode', icon: Play },
  ],
  3: [ // Emotion
    { text: 'How am I feeling?', icon: Smile },
    { text: 'Analyze my expression', icon: Activity },
  ],
  4: [ // Sign Language
    { text: 'What sign did I make?', icon: Hand },
    { text: 'Show me medical signs', icon: Info },
  ],
  5: [ // Face Recognition
    { text: 'Who do you see?', icon: Users },
    { text: 'How many people?', icon: Activity },
  ],
}

// Animated Voice Orb Component
function VoiceOrb({ voiceState }) {
  const getOrbState = () => {
    if (voiceState.wake_detected) return { color: '#fbbf24', label: 'Wake Detected', icon: Zap }
    if (voiceState.listening) return { color: '#10b981', label: 'Listening...', icon: Radio }
    if (voiceState.active) return { color: '#3b82f6', label: 'Processing', icon: Loader2 }
    if (voiceState.enabled) return { color: '#06b6d4', label: 'Ready', icon: Mic }
    return { color: '#6b7280', label: 'Disabled', icon: MicOff }
  }

  const state = getOrbState()
  const IconComponent = state.icon

  const pulseAnimation = voiceState.wake_detected
    ? { scale: [1, 1.3, 1], opacity: [0.8, 1, 0.8] }
    : voiceState.listening
    ? { scale: [1, 1.15, 1], opacity: [0.6, 0.9, 0.6] }
    : { scale: [1, 1.05, 1], opacity: [0.4, 0.6, 0.4] }

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      gap: 24,
      padding: 32 
    }}>
      {/* Orb Container */}
      <div style={{ position: 'relative', width: 220, height: 220 }}>
        {/* Outer glow rings */}
        {voiceState.enabled && [1.8, 1.4, 1.1].map((scale, i) => (
          <motion.div
            key={i}
            animate={{
              scale: [scale, scale * 1.12, scale],
              opacity: voiceState.wake_detected ? [0.3, 0.1, 0.3] : [0.15, 0.05, 0.15]
            }}
            transition={{ 
              duration: 2 + i * 0.5, 
              repeat: Infinity, 
              ease: 'easeInOut',
              delay: i * 0.2 
            }}
            style={{
              position: 'absolute',
              inset: `${-(scale - 1) * 110}px`,
              borderRadius: '50%',
              background: `radial-gradient(circle, ${state.color}40, transparent 60%)`,
              pointerEvents: 'none'
            }}
          />
        ))}

        {/* Pulse ring on activity */}
        {(voiceState.wake_detected || voiceState.listening) && (
          <motion.div
            animate={{ 
              scale: [1, 2.5], 
              opacity: [0.6, 0] 
            }}
            transition={{ 
              duration: 1.5, 
              repeat: Infinity, 
              ease: 'easeOut' 
            }}
            style={{
              position: 'absolute',
              inset: 30,
              borderRadius: '50%',
              border: `3px solid ${state.color}`,
              pointerEvents: 'none'
            }}
          />
        )}

        {/* Main Orb */}
        <motion.div
          animate={pulseAnimation}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            position: 'absolute',
            inset: 30,
            borderRadius: '50%',
            background: `radial-gradient(circle at 35% 35%, 
              ${state.color}dd, 
              ${state.color}88 50%, 
              ${state.color}44
            )`,
            boxShadow: `
              0 0 60px ${state.color}88,
              0 0 100px ${state.color}44,
              inset 0 0 40px rgba(255,255,255,0.1),
              inset -10px -10px 30px rgba(0,0,0,0.2)
            `,
            border: `2px solid ${state.color}aa`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: voiceState.enabled ? 'pointer' : 'default'
          }}
        >
          <motion.div
            animate={voiceState.active ? { rotate: 360 } : {}}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <IconComponent 
              size={48} 
              color={voiceState.enabled ? '#000' : '#ffffff66'} 
              strokeWidth={2.5} 
            />
          </motion.div>
        </motion.div>

        {/* Audio level indicator */}
        {voiceState.listening && (
          <div style={{
            position: 'absolute',
            bottom: -20,
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            gap: 4,
            alignItems: 'flex-end',
            height: 24
          }}>
            {[...Array(12)].map((_, i) => {
              const heights = [8, 12, 16, 20, 18, 14, 22, 16, 12, 18, 20, 14]
              return (
                <motion.div
                  key={i}
                  animate={{
                    height: [8, heights[i], 8]
                  }}
                  transition={{
                    duration: 0.3,
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: i * 0.05
                  }}
                  style={{
                    width: 3,
                    background: state.color,
                    borderRadius: 2
                  }}
                />
              )
            })}
          </div>
        )}
      </div>

      {/* Status Label */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          fontSize: 18,
          fontWeight: 600,
          color: state.color,
          textAlign: 'center',
          textTransform: 'uppercase',
          letterSpacing: 2
        }}
      >
        {state.label}
      </motion.div>

      {/* Wake word hint */}
      {voiceState.enabled && !voiceState.listening && !voiceState.wake_detected && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{
            fontSize: 14,
            color: 'var(--text-3)',
            textAlign: 'center'
          }}
        >
          Say <strong style={{ color: 'var(--cyan)' }}>"Hey ZyGlass"</strong> to start
        </motion.div>
      )}
    </div>
  )
}

// Chat Message Component
function ChatMessage({ type, text, timestamp, mode }) {
  const isUser = type === 'command'
  const ModeIcon = mode ? MODE_ICONS[mode] : null

  return (
    <motion.div
      initial={{ opacity: 0, x: isUser ? 20 : -20, y: 10 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      style={{
        display: 'flex',
        gap: 12,
        alignItems: 'flex-start',
        flexDirection: isUser ? 'row-reverse' : 'row',
        marginBottom: 16
      }}
    >
      {/* Avatar */}
      <div style={{
        width: 40,
        height: 40,
        borderRadius: '50%',
        background: isUser ? 'var(--cyan)' : 'var(--purple)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
      }}>
        {isUser ? (
          <Mic size={20} color="#000" strokeWidth={2.5} />
        ) : (
          <Sparkles size={20} color="#000" strokeWidth={2.5} />
        )}
      </div>

      {/* Message Bubble */}
      <div style={{
        maxWidth: '70%',
        background: isUser ? 'var(--surface-2)' : 'var(--surface-3)',
        padding: '12px 16px',
        borderRadius: 16,
        border: `1px solid ${isUser ? 'var(--cyan)33' : 'var(--purple)33'}`,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
      }}>
        <div style={{
          fontSize: 15,
          color: 'var(--text-1)',
          lineHeight: 1.5,
          marginBottom: 6
        }}>
          {text}
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 11,
          color: 'var(--text-3)'
        }}>
          <Clock size={11} />
          {timestamp}
          {ModeIcon && (
            <>
              <span>•</span>
              <ModeIcon size={11} />
            </>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// Command Chip Component
function CommandChip({ command, icon: Icon, onClick }) {
  return (
    <motion.button
      whileHover={{ scale: 1.03, y: -2 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      style={{
        background: 'var(--surface-2)',
        border: '1px solid var(--cyan)44',
        borderRadius: 12,
        padding: '10px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 13,
        color: 'var(--text-2)',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        fontWeight: 500
      }}
    >
      {Icon && <Icon size={14} />}
      {command}
    </motion.button>
  )
}

// Main Component
export default function AIAssistant({ voiceState, toggleVoice, activeMode }) {
  const [chatHistory, setChatHistory] = useState([])
  const [showCommands, setShowCommands] = useState(true)
  const chatEndRef = useRef(null)
  const lastCommandRef = useRef('')
  const lastResponseRef = useRef('')

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory])

  // Update chat when new command arrives - using useEffect with cleanup
  useEffect(() => {
    if (voiceState.last_command && voiceState.last_command !== lastCommandRef.current) {
      const newCommand = voiceState.last_command
      lastCommandRef.current = newCommand
      
      // Use setTimeout to batch the update
      const timer = setTimeout(() => {
        setChatHistory(prev => [...prev, {
          type: 'command',
          text: newCommand,
          timestamp: voiceState.last_ts || new Date().toLocaleTimeString(),
          mode: activeMode
        }])
      }, 0)
      
      return () => clearTimeout(timer)
    }
  }, [voiceState.last_command, voiceState.last_ts, activeMode])

  // Update chat when new response arrives - using useEffect with cleanup
  useEffect(() => {
    if (voiceState.last_response && voiceState.last_response !== lastResponseRef.current) {
      const newResponse = voiceState.last_response
      lastResponseRef.current = newResponse
      
      // Use setTimeout to batch the update
      const timer = setTimeout(() => {
        setChatHistory(prev => [...prev, {
          type: 'response',
          text: newResponse,
          timestamp: new Date().toLocaleTimeString(),
          mode: activeMode
        }])
      }, 0)
      
      return () => clearTimeout(timer)
    }
  }, [voiceState.last_response, activeMode])

  const handleCommandClick = useCallback((command) => {
    // Simulate voice command (in real app, this would trigger the voice system)
    setChatHistory(prev => [...prev, {
      type: 'command',
      text: command,
      timestamp: new Date().toLocaleTimeString(),
      mode: activeMode
    }])
  }, [activeMode])

  const clearHistory = useCallback(() => {
    setChatHistory([])
    lastCommandRef.current = ''
    lastResponseRef.current = ''
  }, [])

  const currentCommands = COMMAND_TEMPLATES[activeMode] || COMMAND_TEMPLATES.general

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 24px',
        background: 'var(--surface-2)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <h2 style={{
            fontSize: 24,
            fontWeight: 700,
            color: 'var(--text-1)',
            marginBottom: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 12
          }}>
            <Sparkles size={28} color="var(--purple)" />
            ZyGlass Voice AI
          </h2>
          <p style={{
            fontSize: 13,
            color: 'var(--text-3)',
            margin: 0
          }}>
            Voice-powered intelligent assistant
          </p>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {chatHistory.length > 0 && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={clearHistory}
              style={{
                background: 'var(--surface-3)',
                border: '1px solid var(--border)',
                borderRadius: 10,
                padding: '8px 14px',
                color: 'var(--text-2)',
                fontSize: 13,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontWeight: 500
              }}
            >
              <RotateCcw size={14} />
              Clear
            </motion.button>
          )}
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleVoice}
            style={{
              background: voiceState.enabled ? 'var(--cyan)' : 'var(--surface-3)',
              border: `1px solid ${voiceState.enabled ? 'var(--cyan)' : 'var(--border)'}`,
              borderRadius: 10,
              padding: '10px 18px',
              color: voiceState.enabled ? '#000' : 'var(--text-2)',
              fontSize: 14,
              cursor: 'pointer',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              boxShadow: voiceState.enabled ? '0 4px 12px var(--cyan)44' : 'none',
              transition: 'all 0.2s ease'
            }}
          >
            {voiceState.enabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            {voiceState.enabled ? 'Voice On' : 'Voice Off'}
          </motion.button>
        </div>
      </div>

      {/* Main Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden'
      }}>
        {/* Left Panel - Voice Orb */}
        <div style={{
          width: 400,
          background: 'var(--surface-1)',
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 24
        }}>
          <VoiceOrb voiceState={voiceState} />
          
          {/* Quick Stats */}
          <div style={{
            marginTop: 32,
            padding: 20,
            background: 'var(--surface-2)',
            borderRadius: 16,
            border: '1px solid var(--border)',
            width: '100%'
          }}>
            <div style={{
              fontSize: 12,
              color: 'var(--text-3)',
              textTransform: 'uppercase',
              letterSpacing: 1,
              marginBottom: 12,
              fontWeight: 600
            }}>
              Voice Status
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-3)' }}>Commands</span>
                <span style={{ color: 'var(--text-1)', fontWeight: 600 }}>
                  {chatHistory.filter(m => m.type === 'command').length}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-3)' }}>Last Active</span>
                <span style={{ color: 'var(--text-1)', fontWeight: 600 }}>
                  {voiceState.last_ts || 'Never'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Chat Interface */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Chat History */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: 24,
            background: 'var(--surface-1)'
          }}>
            {chatHistory.length === 0 ? (
              <div style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 16,
                color: 'var(--text-3)'
              }}>
                <MessageSquare size={64} strokeWidth={1.5} opacity={0.5} />
                <div style={{
                  fontSize: 18,
                  fontWeight: 600,
                  color: 'var(--text-2)'
                }}>
                  No conversation yet
                </div>
                <div style={{ fontSize: 14, textAlign: 'center', maxWidth: 400 }}>
                  {voiceState.enabled 
                    ? 'Say "Hey ZyGlass" followed by your command to start' 
                    : 'Enable voice to start talking with ZyGlass'}
                </div>
              </div>
            ) : (
              <>
                {chatHistory.map((msg, idx) => (
                  <ChatMessage key={idx} {...msg} />
                ))}
                <div ref={chatEndRef} />
              </>
            )}
          </div>

          {/* Command Suggestions */}
          <div style={{
            borderTop: '1px solid var(--border)',
            background: 'var(--surface-2)',
            padding: 20
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 16
            }}>
              <div style={{
                fontSize: 13,
                color: 'var(--text-3)',
                textTransform: 'uppercase',
                letterSpacing: 1,
                fontWeight: 600
              }}>
                Suggested Commands
              </div>
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowCommands(!showCommands)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-3)',
                  cursor: 'pointer',
                  padding: 4
                }}
              >
                <motion.div
                  animate={{ rotate: showCommands ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronDown size={16} />
                </motion.div>
              </motion.button>
            </div>

            <AnimatePresence>
              {showCommands && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 10,
                    overflow: 'hidden'
                  }}
                >
                  {currentCommands.map((cmd, idx) => (
                    <CommandChip
                      key={idx}
                      command={cmd.text}
                      icon={cmd.icon}
                      onClick={() => handleCommandClick(cmd.text)}
                    />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
