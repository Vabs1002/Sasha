"""
Real-Time Conversational Orchestrator for Sasha AI Interviewer
Inspired by Real-Time Conversational AI Commentator (sushant-mishra-dtu)

Key Capabilities:
1. Server-Authoritative FSM (Finite State Machine)
2. Asynchronous AudioMux with Barge-In (Interruption Handling)
3. Zero-Dead-Air Conversational Fillers during Agentic RAG
4. Token-by-token generation with Grok / OpenAI API
"""

import asyncio
import enum
import logging
import os
import re
from typing import AsyncGenerator, Callable, Dict, Any, List, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from interviewer_agent import search_resume_tool

logger = logging.getLogger(__name__)

class SessionState(str, enum.Enum):
    IDLE = "Idle"
    LISTENING = "Listening"
    TOOL_CALLING = "ToolCalling"
    RESPONDING = "Responding"
    DRAINING = "Draining"

# Valid state transitions to guarantee race-free concurrency
ALLOWED_TRANSITIONS = {
    SessionState.IDLE: {SessionState.LISTENING},
    SessionState.LISTENING: {SessionState.TOOL_CALLING, SessionState.RESPONDING, SessionState.IDLE},
    SessionState.TOOL_CALLING: {SessionState.RESPONDING, SessionState.LISTENING},  # barge-in allows -> Listening
    SessionState.RESPONDING: {SessionState.DRAINING, SessionState.LISTENING},       # barge-in allows -> Listening
    SessionState.DRAINING: {SessionState.LISTENING, SessionState.IDLE}
}

class ConversationalStateMachine:
    """Server-authoritative FSM for managing interview turn states safely."""
    def __init__(self):
        self._state = SessionState.IDLE
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> SessionState:
        return self._state

    async def transition_to(self, new_state: SessionState) -> bool:
        async with self._lock:
            allowed = ALLOWED_TRANSITIONS.get(self._state, set())
            if new_state in allowed:
                logger.info(f"FSM State Change: {self._state.value} -> {new_state.value}")
                self._state = new_state
                return True
            logger.warning(f"Illegal transition rejected: {self._state.value} -> {new_state.value}")
            return False

class RealtimeOrchestrator:
    """
    Coordinates streaming Grok generation, Agentic RAG search,
    conversational fillers, and instant barge-in interruption.
    """
    def __init__(self, grok_api_key: Optional[str] = None):
        self.api_key = grok_api_key or os.getenv("GROK_API_KEY")
        self.fsm = ConversationalStateMachine()
        self.interrupt_event = asyncio.Event()
        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=self.api_key or "missing_key"
        ) if self.api_key else None

        # Fillers to eliminate dead air during RAG retrieval
        self.fillers = [
            "Let me check your resume on that...",
            "Got it, looking at your project details...",
            "Right, verifying that in your background..."
        ]
        self._filler_idx = 0

    def trigger_barge_in(self):
        """Called immediately when user starts speaking during AI output."""
        logger.info("Barge-in triggered! Signaling cancellation of current output.")
        self.interrupt_event.set()

    def _get_next_filler(self) -> str:
        filler = self.fillers[self._filler_idx % len(self.fillers)]
        self._filler_idx += 1
        return filler

    async def process_turn(
        self,
        candidate_utterance: str,
        resume_text: str,
        system_prompt: str = "You are a senior technical interviewer."
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a candidate's turn in a streaming, interruptible pipeline.
        Yields events:
        - {"type": "filler", "text": "..."}
        - {"type": "tool_call", "query": "..."}
        - {"type": "token", "text": "..."}
        - {"type": "interrupted"}
        - {"type": "done"}
        """
        # Check if already interrupted
        if self.interrupt_event.is_set():
            await self.fsm.transition_to(SessionState.LISTENING)
            yield {"type": "interrupted"}
            return

        await self.fsm.transition_to(SessionState.LISTENING)

        # 1. Inspect if Agentic RAG is warranted
        needs_retrieval = any(k in candidate_utterance.lower() for k in ["built", "led", "scaled", "project", "percent", "%", "database", "microservices", "pipeline", "team"])

        retrieved_context = []
        if needs_retrieval and resume_text:
            await self.fsm.transition_to(SessionState.TOOL_CALLING)
            
            # Emit zero-dead-air conversational filler immediately
            filler_text = self._get_next_filler()
            yield {"type": "filler", "text": filler_text}

            # Run Hybrid RRF tool asynchronously (via to_thread to avoid blocking event loop)
            query = candidate_utterance[:100]
            yield {"type": "tool_call", "query": query}
            retrieved_context = await asyncio.to_thread(search_resume_tool, query, resume_text)

            if self.interrupt_event.is_set():
                await self.fsm.transition_to(SessionState.LISTENING)
                yield {"type": "interrupted"}
                return

        # 2. Enter Responding state
        await self.fsm.transition_to(SessionState.RESPONDING)

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Candidate Answer: {candidate_utterance}\nResume Context: {retrieved_context}\nProvide a natural, probing follow-up question."}
        ]

        # 3. Stream Grok response or fallback
        try:
            if not self.client or not self.api_key:
                raise RuntimeError("Grok client not configured")

            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=prompt_messages,
                temperature=0.3,
                stream=True
            )

            for chunk in response:
                if self.interrupt_event.is_set():
                    logger.info("Output interrupted mid-stream by candidate.")
                    await self.fsm.transition_to(SessionState.LISTENING)
                    yield {"type": "interrupted"}
                    return

                token = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
                if token:
                    yield {"type": "token", "text": token}
                    # Minimal yield sleep to allow cooperative concurrency
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.warning(f"Live Grok streaming failed or uncredited ({e}). Yielding fallback stream.")
            # Graceful token streaming fallback so developer can test pipeline locally
            fallback_text = (
                f"Thank you for sharing that. Based on your background with {retrieved_context[0] if retrieved_context else 'these systems'}, "
                f"could you walk me through the trade-offs and how you validated the outcome?"
            )
            for word in fallback_text.split(" "):
                if self.interrupt_event.is_set():
                    await self.fsm.transition_to(SessionState.LISTENING)
                    yield {"type": "interrupted"}
                    return
                yield {"type": "token", "text": word + " "}
                await asyncio.sleep(0.05)

        # 4. Drain and return to Listening for next candidate turn
        await self.fsm.transition_to(SessionState.DRAINING)
        await asyncio.sleep(0.05)
        await self.fsm.transition_to(SessionState.LISTENING)
        yield {"type": "done"}
