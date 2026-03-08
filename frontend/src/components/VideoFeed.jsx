/**
 * VideoFeed — canvas-based MJPEG replacement.
 *
 * Instead of a browser-buffered <img src="...mjpeg">, we poll /snapshot
 * on every requestAnimationFrame tick. Because /snapshot always returns
 * the LATEST frame from the backend, the browser can never build up a
 * backlog — no more stutter / sliding-window lag.
 *
 * Performance profile:
 *  • RAF fires at monitor refresh rate (60 fps)
 *  • A new fetch starts only when the previous one completes (busy gate)
 *  • Actual display FPS = min(ESP32 fps, network throughput)
 *  • Canvas draw is hardware-accelerated — zero CPU composite cost
 */
import { useRef, useEffect, useState, useCallback } from 'react'
import { BASE } from '../api'
import { Maximize2, VideoOff, Wifi, WifiOff } from 'lucide-react'

export default function VideoFeed({ active = true, onFullscreen, className = '' }) {
  const canvasRef   = useRef(null)
  const aliveRef    = useRef(false)
  const busyRef     = useRef(false)
  const rafRef      = useRef(0)
  const frameCountRef = useRef(0)
  const lastFpsRef    = useRef(performance.now())
  const lastTickRef   = useRef(0)   // timestamp of last fetch start — used for 20fps throttle

  const [fps,    setFps]    = useState(0)
  const [online, setOnline] = useState(true)
  // Refs to track state inside RAF closure without stale values
  const onlineRef  = useRef(true)
  const prevFpsRef = useRef(0)

  // ── Canvas resize to parent container ──────────────────────────────────────
  const resizeCanvas = useCallback(() => {
    const c = canvasRef.current
    if (!c) return
    const { width, height } = c.getBoundingClientRect()
    if (c.width !== width || c.height !== height) {
      c.width  = width  || 640
      c.height = height || 480
    }
  }, [])

  useEffect(() => {
    resizeCanvas()
    const ro = new ResizeObserver(resizeCanvas)
    if (canvasRef.current) ro.observe(canvasRef.current)
    return () => ro.disconnect()
  }, [resizeCanvas])

  // ── Streaming loop ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!active) {
      cancelAnimationFrame(rafRef.current)
      aliveRef.current = false
      // Reset error state so it shows correctly when stream resumes
      onlineRef.current = true
      prevFpsRef.current = 0
      setOnline(true)
      setFps(0)
      return
    }

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true })
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'medium'

    aliveRef.current  = true
    busyRef.current   = false
    onlineRef.current = true
    let errStreak     = 0

    const tick = () => {
      if (!aliveRef.current) return
      rafRef.current = requestAnimationFrame(tick)

      // Throttle to 30 fps max — smoother streaming with better responsiveness.
      // At 30fps the server has 33ms per frame to draw overlays and encode JPEG.
      const _now = performance.now()
      if (_now - lastTickRef.current < 33) return
      if (busyRef.current) return
      lastTickRef.current = _now
      busyRef.current = true

      fetch(`${BASE}/snapshot`, {
        cache:   'no-store',
        headers: { Pragma: 'no-cache', 'Cache-Control': 'no-cache' },
      })
        .then(r => {
          if (!r.ok) throw new Error('bad')
          return r.blob()
        })
        .then(blob => createImageBitmap(blob))
        .then(bmp => {
          if (!aliveRef.current) { bmp.close(); return }

          // Fill canvas, center-crop to maintain aspect ratio
          const { width: cw, height: ch } = canvas
          const { width: bw, height: bh } = bmp
          const scale   = Math.max(cw / bw, ch / bh)
          const dx      = (cw - bw * scale) / 2
          const dy      = (ch - bh * scale) / 2
          ctx.drawImage(bmp, dx, dy, bw * scale, bh * scale)
          bmp.close()

          // FPS counter — only call setFps when value actually changes (avoids re-render every frame)
          frameCountRef.current++
          const now = performance.now()
          if (now - lastFpsRef.current >= 900) {
            const elapsed = now - lastFpsRef.current
            const newFps  = Math.round(frameCountRef.current * 1000 / elapsed)
            if (newFps !== prevFpsRef.current) {
              prevFpsRef.current = newFps
              setFps(newFps)
            }
            frameCountRef.current = 0
            lastFpsRef.current    = now
          }

          // Only update React state when online status actually changes
          errStreak = 0
          if (!onlineRef.current) {
            onlineRef.current = true
            setOnline(true)
          }
        })
        .catch(() => {
          errStreak++
          if (errStreak >= 6 && onlineRef.current) {
            onlineRef.current = false
            setOnline(false)
          }
        })
        .finally(() => { busyRef.current = false })
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      aliveRef.current = false
      cancelAnimationFrame(rafRef.current)
    }
  }, [active])

  return (
    <div className={`relative overflow-hidden rounded-xl bg-black ${className}`}
         style={{ containerType: 'size' }}>

      {/* Live canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: active ? 'block' : 'none' }}
      />

      {/* Scanline overlay */}
      <div className="scanlines absolute inset-0 z-10 pointer-events-none" />

      {/* Offline placeholder */}
      {!active && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-[var(--text-3)]">
          <VideoOff size={44} strokeWidth={1.2} />
          <p className="text-xs uppercase tracking-widest">Stream Paused</p>
        </div>
      )}

      {/* Status strip — top-left */}
      {active && (
        <div className="absolute top-2.5 left-2.5 z-20 flex items-center gap-1.5
                        px-2 py-1 rounded-md bg-black/55 border border-white/8
                        text-[11px] font-mono">
          {online
            ? <Wifi      size={10} className="text-[var(--green)]" />
            : <WifiOff   size={10} className="text-[var(--red)]"   />}
          <span className={online ? 'text-[var(--green)]' : 'text-[var(--red)]'}>
            {online ? `${fps} fps` : 'offline'}
          </span>
        </div>
      )}

      {/* Fullscreen button — top-right */}
      <button
        onClick={onFullscreen}
        className="absolute top-2.5 right-2.5 z-20 p-1.5 rounded-lg
                   bg-black/50 hover:bg-black/80 border border-white/10
                   text-[var(--text-2)] hover:text-white transition-all duration-150"
        title="Fullscreen"
      >
        <Maximize2 size={13} strokeWidth={1.8} />
      </button>

      {/* Corner accents — purely decorative */}
      <span className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-[var(--cyan)] rounded-tl-xl z-20 pointer-events-none" />
      <span className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-[var(--cyan)] rounded-tr-xl z-20 pointer-events-none" />
      <span className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-[var(--cyan)] rounded-bl-xl z-20 pointer-events-none" />
      <span className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-[var(--cyan)] rounded-br-xl z-20 pointer-events-none" />
    </div>
  )
}
