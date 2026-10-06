/**
 * Sasha API Client
 * Coordinates with the FastAPI backend (http://localhost:8000)
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    console.warn('API health check error:', err);
    return null;
  }
}

export async function startInterview(resumeFile, jdText = '') {
  const formData = new FormData();
  formData.append('resume_file', resumeFile);
  if (jdText && jdText.trim()) {
    formData.append('jd_text', jdText.trim());
  }

  const res = await fetch(`${API_BASE}/api/v1/interview/start`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to initialize interview session.');
  }

  return await res.json();
}

export async function submitTurn(sessionId, answerText) {
  const res = await fetch(`${API_BASE}/api/v1/interview/${sessionId}/turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer: answerText }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit answer.');
  }

  return await res.json();
}

export async function getSessionStatus(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/interview/${sessionId}/status`);
  if (!res.ok) throw new Error('Failed to retrieve session status');
  return await res.json();
}

export function getReportDownloadUrl(sessionId) {
  return `${API_BASE}/api/v1/interview/${sessionId}/report`;
}

export async function getReportHtml(sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/interview/${sessionId}/report/html`);
  if (!res.ok) throw new Error('HTML report not ready');
  return await res.text();
}

export function createInterviewWebSocket(sessionId) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = API_BASE.replace(/^https?:\/\//, '');
  return new WebSocket(`${protocol}//${host}/ws/interview/${sessionId}`);
}
