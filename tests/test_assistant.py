"""Offline regression tests: real graphs and tools with a deterministic chat model."""
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter" / "src"))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field, ValidationError

from assistant import DocumentAssistant
from schemas import AnswerResponse, UserIntent, SessionState, DocumentChunk, UpdateMemoryResponse, SummarizationResponse, CalculationResponse
from agent import _run_task
from tools import ToolLogger, create_calculator_tool, create_document_search_tool
from retrieval import SimulatedRetriever
from prompts import get_chat_prompt_template


class ScriptedModel(BaseChatModel):
    """Exercise the real ReAct loop, including tool execution and structured output."""
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(value):
            self.seen.append((schema.__name__, value))
            if schema is UserIntent:
                query = str(value).split("User Input: ")[-1].split("\n")[0].lower()
                intent = "calculation" if "calculate" in query else "summarization" if "summarize" in query else "unknown" if "unclear" in query else "qa"
                return schema(intent_type=intent, confidence=.95, reasoning="Scripted offline classification")
            if schema is UpdateMemoryResponse:
                return schema(summary="CON-001: Healthcare Partners LLC, value 180000, monthly fee 15000.", document_ids=["CON-001"])
            if schema is AnswerResponse:
                return schema(question="Who is the contract client?", answer="Healthcare Partners LLC (CON-001).", sources=["CON-001"], confidence=.95)
            if schema.__name__ == "SummarizationResponse":
                return schema(original_length=100, summary="CON-001 covers 12 months for Healthcare Partners LLC.", key_points=["Monthly fee: 15000"], document_ids=["CON-001"])
            return schema(expression="180000 / 12", result=15000, explanation="180000 / 12 = 15000 per month (CON-001).", units="USD/month")
        return RunnableLambda(respond)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = messages[last_human:]
        query = turn[0].content.lower()
        calls = [("document_search", {"query": "CON-001"}), ("document_reader", {"doc_id": "CON-001"})]
        if "calculate" in query:
            calls.append(("calculator", {"expression": "180000 / 12"}))
        completed = sum(isinstance(m, ToolMessage) for m in turn)
        if completed < len(calls):
            name, args = calls[completed]
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(uuid.uuid4())}])
        else:
            message = AIMessage(content="Unstructured draft; the validated answer is returned separately.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def assistant(self):
        assistant = DocumentAssistant("offline-key", session_storage_path=str(self.root / "sessions"), logs_dir=str(self.root / "logs"))
        assistant.llm = ScriptedModel()
        return assistant

    def test_schema_constraints_and_independent_defaults(self):
        for confidence in [-.1, 1.1]:
            with self.assertRaises(ValidationError):
                AnswerResponse(question="q", answer="a", confidence=confidence)
        for invalid in [dict(intent_type="invalid", confidence=.5, reasoning="x"), dict(intent_type="qa", confidence=2, reasoning="x"), dict(intent_type="qa", confidence=.5, reasoning=123)]:
            with self.assertRaises(ValidationError):
                UserIntent(**invalid)
        first = DocumentChunk(doc_id="a", content="a")
        second = DocumentChunk(doc_id="b", content="b")
        first.metadata["test"] = True
        self.assertEqual(second.metadata, {})
        self.assertEqual(SessionState(session_id="x", user_id="u").conversation_history, [])
        self.assertEqual(UpdateMemoryResponse(summary="x").document_ids, [])

    def test_calculator_errors_safety_and_logs(self):
        logger = ToolLogger(str(self.root), "calculator-test")
        calculator = create_calculator_tool(logger)
        for expression, expected in [("2 + 3", "5"), ("(5000 + 12500) * 0.10", "1750.0"), ("2 ** -1", "0.5")]:
            self.assertEqual(calculator.invoke({"expression": expression}), expected)
        for expression in ["1 / 0", "__import__('os')", "9 ** 999999999", "(-1) ** 0.5", "1" * 300]:
            result = calculator.invoke({"expression": expression})
            self.assertTrue(result.startswith(("Error", "Invalid")), result)
        entries = json.loads(Path(logger.log_file).read_text())
        self.assertEqual(len(entries), 8)
        self.assertTrue(all(entry["session_id"] == "calculator-test" for entry in entries))

    def test_search_all_and_id_lookup(self):
        search = create_document_search_tool(SimulatedRetriever(), ToolLogger(str(self.root)))
        for query in ["", "all documents"]:
            self.assertIn("Found 5 document(s)", search.invoke({"query": query, "search_type": "all"}))
        self.assertIn("CON-001", search.invoke({"query": "CON-001"}))

    def test_prompt_selection(self):
        for intent, marker in [("qa", "answering questions"), ("summarization", "summarizer"), ("calculation", "calculation assistant"), ("unknown", "answering questions")]:
            messages = get_chat_prompt_template(intent).format_messages(input="test", chat_history=[])
            self.assertIn(marker, messages[0].content)
            self.assertEqual(messages[-1].content, "test")

    def test_all_routes_real_tools_and_per_turn_actions(self):
        assistant = self.assistant()
        session_id = assistant.start_session("tester")
        self.assertTrue((self.root / "sessions" / f"{session_id}.json").exists())
        for query, intent in [("Who is the client on CON-001?", "qa"), ("Summarize CON-001", "summarization"), ("Calculate the monthly fee for CON-001", "calculation"), ("unclear", "unknown")]:
            result = assistant.process_message(query)
            self.assertTrue(result["success"], result)
            route = "qa" if intent == "unknown" else intent
            self.assertEqual(result["intent"]["intent_type"], intent)
            self.assertEqual(result["actions_taken"], ["classify_intent", f"{route}_agent", "update_memory"])
            expected = ["document_search", "document_reader"] + (["calculator"] if intent == "calculation" else [])
            self.assertEqual(result["tools_used"], expected)
            self.assertIn("CON-001", result["sources"])
            self.assertNotIn("Unstructured draft", result["response"])
        history = assistant._get_conversation_history()
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in history), 4)
        self.assertFalse(any(m.type == "system" for m in history))
        self.assertEqual(len(json.loads(Path(assistant.tool_logger.log_file).read_text())), 9)

    def test_resume_memory_logs_and_session_isolation(self):
        assistant = self.assistant()
        sid = assistant.start_session("tester")
        self.assertTrue(assistant.process_message("Who is the client on CON-001?")["success"])
        resumed = self.assistant()
        resumed.start_session("tester", sid)
        self.assertIn("Healthcare Partners", resumed._get_conversation_summary())
        self.assertEqual(len(resumed.tool_logger.logs), 2)
        result = resumed.process_message("Calculate its monthly fee")
        self.assertTrue(result["success"], result)
        self.assertEqual(len(resumed.tool_logger.logs), 5)
        classification_input = resumed.llm.seen[0][1]
        self.assertIn("Who is the client on CON-001?", classification_input)
        self.assertIn("Healthcare Partners", classification_input)
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in resumed._get_conversation_history()), 2)
        resumed.start_session("another-user")
        self.assertEqual(resumed._get_conversation_history(), [])
        self.assertEqual(resumed.tool_logger.logs, [])

    def test_load_original_session_format(self):
        assistant = self.assistant()
        source = Path(__file__).resolve().parents[1] / "sessions" / "0323d7b0-76e6-44f7-98f1-fedb39614438.json"
        (self.root / "sessions" / source.name).write_text(source.read_text(), encoding="utf-8")
        assistant.start_session("tester", source.stem)
        self.assertTrue(assistant._get_conversation_history())
        self.assertIn("document", assistant._get_conversation_summary().lower())

    def test_errors_do_not_save_successful_turn(self):
        assistant = self.assistant()
        with self.assertRaisesRegex(ValueError, "No active session"):
            assistant.process_message("test")
        assistant.start_session("tester")
        with patch("agent.invoke_react_agent", side_effect=RuntimeError("model unavailable")):
            result = assistant.process_message("test")
        self.assertFalse(result["success"])
        self.assertIn("model unavailable", result["error"])
        self.assertEqual(assistant.current_session.conversation_history, [])

    def test_invalid_structured_answer_is_rejected(self):
        assistant = self.assistant()
        assistant.start_session("tester")
        invalid = {
            "messages": [],
            "structured_response": {"question": "q", "answer": "a", "confidence": 2},
        }
        with patch("agent.invoke_react_agent", return_value=(invalid, [])):
            result = assistant.process_message("Who is the client?")
        self.assertFalse(result["success"])
        self.assertIn("confidence", result["error"])
        self.assertEqual(assistant.current_session.conversation_history, [])

    def test_model_timestamps_are_replaced_with_generation_time(self):
        old = datetime(2023, 10, 6, tzinfo=timezone.utc)
        responses = [
            ("qa", AnswerResponse(question="q", answer="a", confidence=.9, timestamp=old)),
            ("summarization", SummarizationResponse(original_length=1, summary="s", key_points=[], timestamp=old)),
            ("calculation", CalculationResponse(expression="2+3", result=5, explanation="5", timestamp=old)),
        ]
        for intent, response in responses:
            with self.subTest(intent=intent):
                before = datetime.now(timezone.utc)
                with patch("agent.invoke_react_agent", return_value=({"messages": [], "structured_response": response}, [])):
                    result = _run_task({"user_input": "test"}, {"configurable": {"llm": None, "tools": []}}, intent, type(response))
                actual = datetime.fromisoformat(result["current_response"]["timestamp"].replace("Z", "+00:00"))
                self.assertGreaterEqual(actual, before)
                self.assertLessEqual(actual, datetime.now(timezone.utc))


if __name__ == "__main__":
    unittest.main()
