import React, { useEffect, useRef, useState } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { api } from './api.js'

// --- Login ---

function Login({ onLoggedIn }) {
  const [email, setEmail] = useState('alice@example.com')
  const [password, setPassword] = useState('customer-pass')
  const [error, setError] = useState('')
  const nav = useNavigate()

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    try {
      await api.login(email, password)
      const me = await api.me()
      onLoggedIn(me)
      nav('/chat')
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="auth-page">
      <form onSubmit={submit} className="card">
        <h1>Insurance Agent</h1>
        <label>
          Email
          <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" required />
        </label>
        <label>
          Password
          <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" required />
        </label>
        {error && <div className="error">{error}</div>}
        <button type="submit">Sign in</button>
        <p className="hint">
          Try: alice@example.com / customer-pass · admin@example.com / admin-change-me
        </p>
      </form>
    </div>
  )
}

// --- Chat ---

function Chat({ me, onLogout }) {
  const nav = useNavigate()
  const [sessions, setSessions] = useState([])
  const [activeId, setActiveId] = useState(null)
  // Keyed by session id, so a reply that's still streaming keeps updating
  // its own session's state even if the user switches to another chat.
  // Nothing gets silently dropped when you're not looking at it.
  const [messagesBySession, setMessagesBySession] = useState({})
  const [streamingSessions, setStreamingSessions] = useState(new Set())
  const [sourcesBySession, setSourcesBySession] = useState({})
  const [toolsBySession, setToolsBySession] = useState({})
  const [updatedSessions, setUpdatedSessions] = useState(new Set())
  const [input, setInput] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editingTitle, setEditingTitle] = useState('')

  // Tracks the latest activeId so the async send() below can tell, once a
  // reply finishes, whether the user has since switched to another chat.
  const activeIdRef = useRef(activeId)
  useEffect(() => { activeIdRef.current = activeId }, [activeId])

  const messages = messagesBySession[activeId] || []
  const streaming = streamingSessions.has(activeId)
  const sources = sourcesBySession[activeId] || []
  const tools = toolsBySession[activeId] || []

  useEffect(() => { api.listSessions().then(setSessions) }, [])

  const openSession = async (id) => {
    setActiveId(id)
    setUpdatedSessions((s) => { const next = new Set(s); next.delete(id); return next })
    if (!messagesBySession[id]) {
      const msgs = await api.listMessages(id)
      setMessagesBySession((m) => ({ ...m, [id]: msgs }))
    }
  }

  const newSession = async () => {
    const s = await api.createSession('New conversation')
    setSessions((prev) => [s, ...prev])
    setMessagesBySession((m) => ({ ...m, [s.id]: [] }))
    setActiveId(s.id)
  }

  const send = async () => {
    const sessionId = activeId
    if (!sessionId || !input.trim() || streamingSessions.has(sessionId)) return
    const question = input
    setInput('')

    const userMsg = { role: 'user', content: question, id: Date.now() + '-u' }
    const assistantId = Date.now() + '-a'
    setMessagesBySession((m) => ({
      ...m,
      [sessionId]: [...(m[sessionId] || []), userMsg, { role: 'assistant', content: '', pending: true, id: assistantId }],
    }))
    setStreamingSessions((s) => new Set(s).add(sessionId))
    setSourcesBySession((m) => ({ ...m, [sessionId]: [] }))
    setToolsBySession((m) => ({ ...m, [sessionId]: [] }))

    const updateAssistant = (patch) => {
      setMessagesBySession((m) => ({
        ...m,
        [sessionId]: (m[sessionId] || []).map((msg) =>
          msg.id === assistantId ? { ...msg, ...patch } : msg
        ),
      }))
    }

    let assistantContent = ''
    try {
      for await (const evt of api.streamMessage(sessionId, question)) {
        if (evt.event === 'routing') setToolsBySession((m) => ({ ...m, [sessionId]: evt.data.tools || [] }))
        else if (evt.event === 'sources') setSourcesBySession((m) => ({ ...m, [sessionId]: evt.data || [] }))
        else if (evt.event === 'answer_chunk') {
          assistantContent += evt.data.delta
          updateAssistant({ content: assistantContent, pending: false })
        } else if (evt.event === 'awaiting_approval') {
          updateAssistant({ content: `Awaiting approval: ${evt.data.summary}`, pending: false })
        }
      }
    } catch (e) {
      updateAssistant({ content: `Error: ${e.message}`, pending: false })
    } finally {
      setStreamingSessions((s) => { const next = new Set(s); next.delete(sessionId); return next })
      if (activeIdRef.current !== sessionId) {
        setUpdatedSessions((s) => new Set(s).add(sessionId))
      }
      // The backend auto-titles a session from its first message and bumps
      // its updated_at; refresh so the sidebar reflects both.
      api.listSessions().then(setSessions)
    }
  }

  const startRename = (s) => {
    setEditingId(s.id)
    setEditingTitle(s.title)
  }

  const commitRename = async () => {
    const id = editingId
    const title = editingTitle.trim()
    setEditingId(null)
    if (!id || !title) return
    const current = sessions.find((s) => s.id === id)
    if (!current || current.title === title) return
    const updated = await api.renameSession(id, title)
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title: updated.title } : s)))
  }

  const logout = async () => {
    await api.logout().catch(() => {})
    onLogout()
    nav('/login')
  }

  const deleteSession = async (id) => {
    if (!window.confirm('Delete this chat? This cannot be undone.')) return
    await api.deleteSession(id)
    setSessions((prev) => prev.filter((s) => s.id !== id))
    const dropKey = (m) => { const next = { ...m }; delete next[id]; return next }
    setMessagesBySession(dropKey)
    setSourcesBySession(dropKey)
    setToolsBySession(dropKey)
    setUpdatedSessions((s) => { const next = new Set(s); next.delete(id); return next })
    if (activeId === id) setActiveId(null)
  }

  return (
    <div className="chat-page">
      <aside>
        <div className="user-badge">
          <button className="logout-btn" onClick={logout}>
            <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M7.5 17.5H4.5C3.94772 17.5 3.5 17.0523 3.5 16.5V3.5C3.5 2.94772 3.94772 2.5 4.5 2.5H7.5"
                stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M13.3333 14.1667L17.5 10L13.3333 5.83334" stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" />
              <path d="M17.5 10H7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Log out
          </button>
          <strong>{me.full_name}</strong>
          <span>{me.user_type}{me.department ? ` · ${me.department}` : ''}</span>
        </div>
        {me.user_type !== 'customer' && (
          <button className="secondary" onClick={() => nav('/approvals')}>Pending approvals</button>
        )}
        <button className="primary" onClick={newSession}>+ New chat</button>
        <ul className="sessions">
          {sessions.map((s) => (
            <li
              key={s.id}
              className={[
                s.id === activeId ? 'active' : '',
                updatedSessions.has(s.id) ? 'has-update' : '',
              ].filter(Boolean).join(' ')}
              onClick={() => openSession(s.id)}
            >
              {editingId === s.id ? (
                <input
                  className="session-rename-input"
                  value={editingTitle}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setEditingTitle(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { e.preventDefault(); commitRename() }
                    else if (e.key === 'Escape') { e.preventDefault(); setEditingId(null) }
                  }}
                />
              ) : (
                <>
                  <span className="session-title" onDoubleClick={() => startRename(s)}>{s.title}</span>
                  <button
                    className="session-delete-btn"
                    title="Delete chat"
                    aria-label="Delete chat"
                    onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}
                  >
                    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                      <path d="M4 6H16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                      <path d="M8 6V4.5C8 3.94772 8.44772 3.5 9 3.5H11C11.5523 3.5 12 3.94772 12 4.5V6"
                        stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M5.5 6L6.16667 16.5C6.16667 17.0523 6.61438 17.5 7.16667 17.5H12.8333C13.3856 17.5 13.8333 17.0523 13.8333 16.5L14.5 6"
                        stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </>
              )}
            </li>
          ))}
        </ul>
      </aside>
      <main>
        <div className="messages">
          {messages.map((m) => (
            <div key={m.id} className={`msg msg-${m.role}`}>
              <div className="role">{m.role}</div>
              <div className="content">
                {m.pending && !m.content
                  ? <span className="typing-dots"><span></span><span></span><span></span></span>
                  : m.content}
              </div>
            </div>
          ))}
        </div>
        {tools.length > 0 && (
          <div className="meta">Tools used: {tools.join(', ')}</div>
        )}
        {sources.length > 0 && (
          <details className="meta">
            <summary>{sources.length} sources</summary>
            <ul>{sources.map((s) => <li key={s.id}><code>{s.id}</code> {s.content.slice(0, 200)}…</li>)}</ul>
          </details>
        )}
        <div className="composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={activeId ? 'Ask about your policies or coverage…' : 'Create a new chat to start'}
            disabled={!activeId || streaming}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
            }}
          />
          <button onClick={send} disabled={!activeId || streaming}>
            {streaming ? 'Thinking…' : 'Send'}
          </button>
        </div>
      </main>
    </div>
  )
}

// --- Approvals ---

const ACTION_LABELS = {
  file_claim: 'File claim',
  request_policy_cancellation: 'Cancel policy',
  request_policy_change: 'Change coverage',
  document_deletion: 'Delete document',
}

function Approvals({ onLogout }) {
  const nav = useNavigate()
  const [approvals, setApprovals] = useState([])
  const [busyId, setBusyId] = useState(null)
  const [noteById, setNoteById] = useState({})
  const [error, setError] = useState('')

  const refresh = () => api.listApprovals('pending').then(setApprovals).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  const decide = async (id, decision) => {
    setBusyId(id)
    setError('')
    try {
      await api.decideApproval(id, decision, noteById[id] || undefined)
      setApprovals((prev) => prev.filter((a) => a.id !== id))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusyId(null)
    }
  }

  const logout = async () => {
    await api.logout().catch(() => {})
    onLogout()
    nav('/login')
  }

  return (
    <div className="approvals-page">
      <div className="approvals-header">
        <h1>Pending approvals</h1>
        <div>
          <button onClick={() => nav('/chat')}>Back to chat</button>
          <button className="logout-btn" onClick={logout}>Log out</button>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {approvals.length === 0 ? (
        <p className="hint">Nothing waiting on you right now.</p>
      ) : (
        <ul className="approvals-list">
          {approvals.map((a) => (
            <li key={a.id} className="approval-card">
              <div className="approval-type">{ACTION_LABELS[a.action_type] || a.action_type}</div>
              <ul className="approval-args">
                {Object.entries(a.args).map(([k, v]) => (
                  <li key={k}><code>{k}</code>: {String(v)}</li>
                ))}
              </ul>
              <div className="meta">Requested {new Date(a.created_at).toLocaleString()}</div>
              <input
                placeholder="Optional note"
                value={noteById[a.id] || ''}
                onChange={(e) => setNoteById((m) => ({ ...m, [a.id]: e.target.value }))}
              />
              <div className="approval-actions">
                <button disabled={busyId === a.id} onClick={() => decide(a.id, 'approve')}>Approve</button>
                <button disabled={busyId === a.id} onClick={() => decide(a.id, 'reject')}>Reject</button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// --- App shell ---

export default function App() {
  const [me, setMe] = useState(null)
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    (async () => {
      // Redeems the httpOnly refresh cookie (if any) for a fresh access
      // token, so a page reload doesn't force a re-login.
      if (await api.init()) {
        await api.me().then(setMe).catch(() => {})
      }
      setChecking(false)
    })()
  }, [])

  if (checking) return null

  return (
    <Routes>
      <Route path="/login" element={<Login onLoggedIn={setMe} />} />
      <Route
        path="/chat"
        element={me ? <Chat me={me} onLogout={() => setMe(null)} /> : <Navigate to="/login" replace />}
      />
      <Route
        path="/approvals"
        element={
          !me ? <Navigate to="/login" replace />
          : me.user_type === 'customer' ? <Navigate to="/chat" replace />
          : <Approvals onLogout={() => setMe(null)} />
        }
      />
      <Route path="*" element={<Navigate to={me ? '/chat' : '/login'} replace />} />
    </Routes>
  )
}
