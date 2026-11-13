import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_openai import ChatOpenAI
from langgraph.types import Overwrite

from schemas import SessionState
from retrieval import SimulatedRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow


class DocumentAssistant:
    """Route document tasks and persist validated responses and conversation memory."""

    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o",
                 temperature: float = 0.1, session_storage_path: str = "./sessions",
                 logs_dir: str = "./logs", base_url: Optional[str] = None):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model_name, temperature=temperature,
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        )
        self.retriever = SimulatedRetriever()
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        self.tool_logger = None
        self.tools = []
        self.workflow = create_workflow(self.llm, self.tools)
        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)
        self.current_session: Optional[SessionState] = None

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Create a session or resume saved memory and append to its existing log."""
        session_id = session_id or str(uuid.uuid4())
        if not session_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in session_id):
            raise ValueError("Session IDs may contain only letters, digits, '-' and '_'.")
        if self._session_exists(session_id):
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            self.current_session = SessionState(session_id=session_id, user_id=user_id)
            print(f"Started new session {session_id}")
        self.tool_logger = ToolLogger(logs_dir=self.logs_dir, session_id=session_id)
        self.tools = get_all_tools(self.retriever, self.tool_logger)
        self._save_session()
        return session_id

    def _session_exists(self, session_id: str) -> bool:
        return os.path.exists(os.path.join(self.session_storage_path, f"{session_id}.json"))

    def _load_session(self, session_id: str) -> SessionState:
        with open(os.path.join(self.session_storage_path, f"{session_id}.json"), encoding="utf-8") as f:
            data = json.load(f)
        # Older files stored cumulative graph state in the final history entry.
        if data.get("conversation_history") and "messages" not in data:
            last_turn = data["conversation_history"][-1]
            data["messages"] = [
                message if "data" in message else {"type": message["type"], "data": message}
                for message in last_turn.get("messages", [])
                if message.get("type") != "system"
            ]
            data["conversation_summary"] = last_turn.get("conversation_summary", "No previous conversation.")
        return SessionState.model_validate(data)

    def _save_session(self) -> None:
        if self.current_session:
            filepath = os.path.join(self.session_storage_path, f"{self.current_session.session_id}.json")
            # Atomic replacement avoids truncating a session if a write is interrupted.
            with open(filepath + ".tmp", "w", encoding="utf-8") as f:
                f.write(self.current_session.model_dump_json(indent=2))
            os.replace(filepath + ".tmp", filepath)

    def _get_conversation_summary(self, config=None) -> str:
        return self.current_session.conversation_summary if self.current_session else "No previous conversation."

    def _get_conversation_history(self, config=None) -> List[BaseMessage]:
        return messages_from_dict(self.current_session.messages) if self.current_session else []

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Restore saved memory, run the graph, and persist a compact turn record."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")
        config = {"configurable": {
            "thread_id": self.current_session.session_id, "llm": self.llm, "tools": self.tools,
        }}
        try:
            history = self._get_conversation_history()
            initial_state = {
                "messages": Overwrite(history),
                "user_input": user_input,
                "intent": None,
                "next_step": "classify_intent",
                "conversation_summary": self._get_conversation_summary(),
                "active_documents": self.current_session.document_context,
                "current_response": None,
                "tools_used": [],
                "session_id": self.current_session.session_id,
                "user_id": self.current_session.user_id,
                "actions_taken": Overwrite([]),
            }
            final_state = self.workflow.invoke(initial_state, config=config)
            payload = final_state["current_response"]
            response = payload.get("answer") or payload.get("summary") or payload.get("explanation")
            intent = final_state["intent"]
            serialized_messages = messages_to_dict(final_state["messages"])
            self.current_session.messages = serialized_messages
            self.current_session.conversation_summary = final_state["conversation_summary"]
            self.current_session.document_context = final_state["active_documents"]
            self.current_session.last_updated = datetime.now()
            self.current_session.conversation_history.append({
                "user_input": user_input,
                "response": payload,
                "intent": intent,
                "messages": serialized_messages[len(history):],
                "tools_used": final_state["tools_used"],
                "actions_taken": final_state["actions_taken"],
                "conversation_summary": final_state["conversation_summary"],
                "active_documents": final_state["active_documents"],
            })
            self._save_session()
            return {
                "success": True, "response": response, "structured_response": payload,
                "intent": intent, "tools_used": final_state["tools_used"],
                "sources": final_state["active_documents"],
                "actions_taken": final_state["actions_taken"],
                "summary": final_state["conversation_summary"],
            }
        except Exception as e:
            return {"success": False, "error": str(e), "response": None}
