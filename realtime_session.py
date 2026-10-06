"""
Interactive Real-Time Conversational Session
Exercises streaming tokens, zero-dead-air fillers, Hybrid RRF RAG, and barge-in interruption.
"""

import asyncio
import sys
import os
from orchestrator import RealtimeOrchestrator

SAMPLE_RESUME = """
Experienced Senior Machine Learning Engineer with 6 years leading AI initiatives.
Designed and scaled a real-time recommendation engine handling 45,000 requests/sec with Redis and Kafka.
Migrated monolithic microservices to Kubernetes on AWS, cutting latency by 35% and infrastructure cost by $120k.
Built automated ETL data pipelines with PySpark and Airflow processing 10TB of daily event logs.
Proficient in Python, PyTorch, FAISS, Docker, Kubernetes, and FastAPI.
"""

async def run_session():
    print("=" * 65)
    print("  SASHA AI INTERVIEWER - REAL-TIME CONVERSATIONAL SESSION")
    print("  (Full-Duplex Streaming + Barge-In + True Hybrid RRF RAG)")
    print("=" * 65)
    print("Type your answer as a candidate. Type 'interrupt' while speaking to test barge-in.")
    print("Type 'exit' to quit.\n")

    orchestrator = RealtimeOrchestrator()

    while True:
        try:
            candidate_text = input("\n[Candidate] > ").strip()
            if not candidate_text:
                continue
            if candidate_text.lower() in ["exit", "quit"]:
                print("Ending session.")
                break

            print("\n[Sasha AI]: ", end="", flush=True)

            async for event in orchestrator.process_turn(candidate_text, SAMPLE_RESUME):
                evt_type = event.get("type")
                if evt_type == "filler":
                    print(f"\n⚡ [Zero-Dead-Air Filler] {event['text']}\n[Sasha AI]: ", end="", flush=True)
                elif evt_type == "tool_call":
                    # Debug notification of Agentic Hybrid RAG
                    pass
                elif evt_type == "token":
                    print(event["text"], end="", flush=True)
                elif evt_type == "interrupted":
                    print("\n🛑 [Interrupted / Barge-in Detected! Sasha yielded turn.]")
                    break
                elif evt_type == "done":
                    print()

        except KeyboardInterrupt:
            print("\nSession stopped.")
            break

if __name__ == "__main__":
    asyncio.run(run_session())
