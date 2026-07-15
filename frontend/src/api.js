// Minimal API client. The access token lives only in memory (short-lived,
// fine to lose on reload). The refresh token lives in an httpOnly cookie
// set by the backend, so JS never touches it directly; `credentials: 'include'`
// is required on every request. On app load, api.init() silently redeems that
// cookie for a fresh access token, so a page refresh doesn't log you out.
// A 401 on any other request triggers one transparent refresh-and-retry.
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

let accessToken = null
export const setToken = (t) => { accessToken = t }
export const getToken = () => accessToken

async function tryRefresh() {
  try {
    const res = await fetch(`${API_BASE}/api/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    if (!res.ok) { accessToken = null; return false }
    const data = await res.json()
    accessToken = data.access_token
    return true
  } catch {
    accessToken = null
    return false
  }
}

// fetch() with the Authorization header and cookie attached, that
// transparently refreshes the access token once on a 401 and retries.
async function authFetch(path, opts = {}, retried = false) {
  const headers = { ...(opts.headers || {}) }
  if (accessToken) headers['Authorization'] = `Bearer ${accessToken}`

  const res = await fetch(`${API_BASE}${path}`, { ...opts, headers, credentials: 'include' })
  if (res.status === 401 && !retried && await tryRefresh()) {
    return authFetch(path, opts, true)
  }
  return res
}

async function request(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) }
  const res = await authFetch(path, { ...opts, headers })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || body.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  // Called once on app load: tries to silently redeem the refresh cookie
  // (if any) for an access token. Returns whether that succeeded.
  init: () => tryRefresh(),
  login: async (email, password) => {
    const data = await request('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) })
    accessToken = data.access_token
    return data
  },
  me: () => request('/api/auth/me'),
  logout: async () => {
    await request('/api/auth/logout', { method: 'POST', body: JSON.stringify({}) }).catch(() => {})
    accessToken = null
  },
  listSessions: () => request('/api/chat/sessions'),
  createSession: (title) => request('/api/chat/sessions', { method: 'POST', body: JSON.stringify({ title }) }),
  renameSession: (sessionId, title) => request(`/api/chat/sessions/${sessionId}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteSession: async (sessionId) => {
    const res = await authFetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`) // 204 No Content has no body to parse
  },
  listMessages: (sessionId) => request(`/api/chat/sessions/${sessionId}/messages`),
  listPolicies: () => request('/api/policies'),

  // SSE stream for chat. Returns EventSource-like async iterator over events.
  streamMessage: async function* (sessionId, content) {
    const res = await authFetch(`/api/chat/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // Parse SSE: events are separated by a blank line. sse-starlette uses
      // \r\n line endings by default, so normalize before splitting.
      const events = buffer.replace(/\r\n/g, '\n').split('\n\n')
      buffer = events.pop() // keep the incomplete last one
      for (const evt of events) {
        const lines = evt.split('\n')
        let event = 'message'
        let data = ''
        for (const line of lines) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) data += line.slice(5).trim()
        }
        if (data) yield { event, data: JSON.parse(data) }
      }
    }
  },
}
