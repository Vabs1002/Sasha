"""
Real-Time Conversational Orchestrator for Sasha AI Interviewer
Inspired by Real-Time Conversational AI Commentator (sushant-mishra-dtu)

Key Capabilities:
1. Server-Authoritative FSM (Finite State Machine)
2. Asynchronous AudioMux with Barge-In (Interruption Handling)
3. Optional conversational fillers during resume retrieval
4. Token-by-token generation with the configured OpenAI-compatible provider
"""

import asyncio
import enum
import logging
import os
import re
import threading
from typing import AsyncGenerator, Callable, Dict, Any, List, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from interviewer_agent import (
    client as configured_client,
    default_llm_model,
    search_resume_tool,
)

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
    Experimental standalone text-streaming demo. The browser interview API
    has its own turn pipeline; this helper is not the app's audio transport.
    """
    def __init__(self, provider_client=None, model: Optional[str] = None, grok_api_key: Optional[str] = None):
        # Default to the same configured provider/model as the interviewer.
        # The old explicit Grok argument remains available for local experiments.
        self.api_key = grok_api_key
        self.fsm = ConversationalStateMachine()
        self.interrupt_event = asyncio.Event()
        self.client = provider_client or configured_client
        self.model = model or default_llm_model
        if grok_api_key:
            self.client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=grok_api_key,
                timeout=15.0,
                max_retries=0,
            )
            self.model = os.getenv("GROK_MODEL", "grok-beta")

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

    async def _stream_completion(self, messages: List[Dict[str, str]]):
        """Yield provider text without blocking the event loop on sync SDK I/O."""
        if not self.client:
            raise RuntimeError("No interview provider is configured")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_stream = threading.Event()

        def produce_chunks():
            response = None
            try:
                options = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "stream": True,
                }
                if self.model.lower().startswith("gemini-"):
                    options["extra_body"] = {
                        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "minimal")
                    }
                response = self.client.chat.completions.create(**options)
                for chunk in response:
                    if stop_stream.is_set():
                        break
                    choices = getattr(chunk, "choices", None) or []
                    delta = getattr(choices[0], "delta", None) if choices else None
                    token = getattr(delta, "content", None) if delta else None
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as error:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error))
            finally:
                close = getattr(response, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass

        producer_task = asyncio.create_task(asyncio.to_thread(produce_chunks))
        while True:
            if self.interrupt_event.is_set():
                stop_stream.set()
                producer_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                yield {"type": "interrupted"}
                return

            queue_task = asyncio.create_task(queue.get())
            interrupt_task = asyncio.create_task(self.interrupt_event.wait())
            done, pending = await asyncio.wait(
                {queue_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_task in done and self.interrupt_event.is_set():
                queue_task.cancel()
                await asyncio.gather(queue_task, return_exceptions=True)
                stop_stream.set()
                producer_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                yield {"type": "interrupted"}
                return
            interrupt_task.cancel()
            await asyncio.gather(interrupt_task, return_exceptions=True)
            kind, payload = queue_task.result()
            if kind == "token":
                yield {"type": "token", "text": payload}
            elif kind == "error":
                raise payload
            else:
                await producer_task
                return

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
        # A pre-signalled interrupt cancels this request once; clear it so the
        # next candidate turn can proceed normally.
        if self.interrupt_event.is_set():
            self.interrupt_event.clear()
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
            retrieval_task = asyncio.create_task(asyncio.to_thread(search_resume_tool, query, resume_text))
            interrupt_task = asyncio.create_task(self.interrupt_event.wait())
            done, pending = await asyncio.wait(
                {retrieval_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_task in done and self.interrupt_event.is_set():
                retrieval_task.cancel()
                await asyncio.gather(retrieval_task, return_exceptions=True)
                self.interrupt_event.clear()
                await self.fsm.transition_to(SessionState.LISTENING)
                yield {"type": "interrupted"}
                return
            interrupt_task.cancel()
            await asyncio.gather(interrupt_task, return_exceptions=True)
            retrieved_context = await retrieval_task

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

        # 3. Stream the configured provider response, or a local fallback.
        try:
            async for event in self._stream_completion(prompt_messages):
                if event["type"] == "interrupted":
                    logger.info("Output interrupted by candidate.")
                    self.interrupt_event.clear()
                    await self.fsm.transition_to(SessionState.LISTENING)
                    yield event
                    return
                yield event

        except Exception as e:
            logger.warning(f"Live Grok streaming failed or uncredited ({e}). Yielding fallback stream.")
            # Graceful token streaming fallback so developer can test pipeline locally
            fallback_text = (
                f"Thank you for sharing that. Based on your background with {retrieved_context[0] if retrieved_context else 'these systems'}, "
                f"could you walk me through the trade-offs and how you validated the outcome?"
            )
            for word in fallback_text.split(" "):
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()
                    await self.fsm.transition_to(SessionState.LISTENING)
                    yield {"type": "interrupted"}
                    return
                yield {"type": "token", "text": word + " "}
                await asyncio.sleep(0)

        # 4. Drain and return to Listening for next candidate turn
        await self.fsm.transition_to(SessionState.DRAINING)
        await asyncio.sleep(0.05)
        await self.fsm.transition_to(SessionState.LISTENING)
        yield {"type": "done"}
