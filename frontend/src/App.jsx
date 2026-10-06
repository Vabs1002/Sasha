import React, { useState, useEffect } from 'react';
import LobbyView from './components/LobbyView';
import InterviewRoom from './components/InterviewRoom';
import ReportView from './components/ReportView';
import { checkHealth, startInterview } from './api';

export default function App() {
  const [currentView, setCurrentView] = useState('lobby'); // 'lobby' | 'room' | 'report'
  const [sessionData, setSessionData] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [serverOnline, setServerOnline] = useState(true);

  // Check backend server status periodically
  useEffect(() => {
    async function verifyBackend() {
      const health = await checkHealth();
      setServerOnline(Boolean(health && health.status === 'healthy'));
    }
    verifyBackend();
    const interval = setInterval(verifyBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleStartSession = async ({ resumeFile, jdText }) => {
    setIsStarting(true);
    try {
      const data = await startInterview(resumeFile, jdText);
      setSessionData(data);
      setCurrentView('room');
    } catch (err) {
      console.error('Failed to start interview:', err);
      alert(err.message || 'Error initializing interview session.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleCompleteSession = (sessionId) => {
    setCurrentView('report');
  };

  const handleRestart = () => {
    setSessionData(null);
    setCurrentView('lobby');
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 grid-bg selection:bg-sky-500/30 selection:text-sky-200">
      {currentView === 'lobby' && (
        <LobbyView
          onStartSession={handleStartSession}
          isStarting={isStarting}
          serverOnline={serverOnline}
        />
      )}

      {currentView === 'room' && sessionData && (
        <InterviewRoom
          sessionData={sessionData}
          onCompleteSession={handleCompleteSession}
        />
      )}

      {currentView === 'report' && sessionData && (
        <ReportView
          sessionId={sessionData.session_id}
          onRestart={handleRestart}
        />
      )}
    </div>
  );
}
