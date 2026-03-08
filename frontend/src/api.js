// ── ZYGLASS API layer ─────────────────────────────────────────────────────
export const BASE = 'http://localhost:5000'

/** GET a JSON endpoint; returns parsed data or throws. */
export async function apiGet(path) {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`[API] GET ${path} → ${r.status}`)
  return r.json()
}

/** POST JSON body; returns parsed response or throws. */
export async function apiPost(path, body = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`[API] POST ${path} → ${r.status}`)
  return r.json()
}

/** Snapshot URL — cache-busted so browser never serves stale frame. */
export const snapshotUrl = () => `${BASE}/snapshot?_=${Date.now()}`

// ── Preset endpoints ────────────────────────────────────────────────────────
export const api = {
  health:       ()        => apiGet('/health'),
  currentMode:  ()        => apiGet('/current_mode'),
  setMode:      (id)      => apiPost(`/set_mode/${id}`),
  ocrData:      ()        => apiGet('/ocr_data'),
  signData:     ()        => apiGet('/sign_data'),
  faceData:     ()        => apiGet('/face_data'),
  voiceData:    ()        => apiGet('/voice_data'),
  lectureData:  ()        => apiGet('/lecture_data'),
  currencyData: ()        => apiGet('/currency_data'),
  dbLog:        ()        => apiGet('/db_log'),
  toggleVoice:  (enable)  => apiPost('/toggle_voice',   { enable }),
  toggleCurrency:(enable) => apiPost('/toggle_currency', { enable }),
  toggleLecture: (enable, clear = false) => apiPost('/toggle_lecture', { enable, clear }),
  enrollUnknown: (name)   => apiPost('/enroll_unknown',  { name }),
}
