"""Run real model conversations and produce session and tool-call evidence."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter" / "src"))
from assistant import DocumentAssistant


def main():
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in .env before running the live demo.")
    assistant = DocumentAssistant(
        key, session_storage_path=str(ROOT / "sessions"), logs_dir=str(ROOT / "logs")
    )
    session_id = assistant.start_session("rubric-demo")
    queries = [
        "Who is the client on contract CON-001?",
        "Summarize contract CON-001, including its services and termination terms.",
        "Calculate the monthly fee from CON-001's total value and duration. Use the calculator tool.",
    ]
    for query, expected in zip(queries, ["qa", "summarization", "calculation"]):
        print(f"\nUSER: {query}", flush=True)
        result = assistant.process_message(query)
        if not result["success"]:
            raise SystemExit(f"Live demo failed: {result['error']}")
        print(json.dumps(result, indent=2), flush=True)
        if result["intent"]["intent_type"] != expected:
            raise SystemExit(f"Expected {expected} intent; inspect the recorded session.")
        if expected == "calculation" and "calculator" not in result["tools_used"]:
            raise SystemExit("Calculation did not use calculator; inspect the recorded session.")
    # Re-create the assistant to verify memory restoration across processes/instances.
    resumed = DocumentAssistant(key, session_storage_path=str(ROOT / "sessions"), logs_dir=str(ROOT / "logs"))
    resumed.start_session("rubric-demo", session_id)
    result = resumed.process_message("What were its termination terms again?")
    if not result["success"]:
        raise SystemExit(f"Resumed conversation failed: {result['error']}")
    print("\nRESUMED FOLLOW-UP:", json.dumps(result, indent=2), flush=True)
    print(f"\nSession evidence: sessions/{session_id}.json")
    print(f"Tool evidence: logs/session_{session_id}.json")


if __name__ == "__main__":
    main()
