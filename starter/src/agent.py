from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
import operator
from datetime import datetime, timezone
# pyrefly: ignore[missing-import]
from schemas import (
    UserIntent,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
# pyright: ignore[reportMissingImports]
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[Dict[str, Any]]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # Actions taken
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> tuple[Dict[str, Any], List[str]]:
    agent = create_react_agent(
        model=llm,
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", [])[len(messages):] if isinstance(t, ToolMessage)]

    return result, tools_used


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = (
        f"Summary: {state.get('conversation_summary', '')}\n"
        f"Active documents: {state.get('active_documents', [])}\n"
        f"Messages: {state.get('messages', [])}"
    )

    # Configure the llm chat model for structured output
    llm_with_structure = llm.with_structured_output(UserIntent)
    # Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt().format(
        user_input=state.get("user_input", ""),
        conversation_history=history,
    )
    intent = llm_with_structure.invoke(prompt)

    # Conditional logic to set next_step based on intent
    if intent.intent_type == "qa":
        next_step = "qa_agent"
    elif intent.intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent.intent_type == "calculation":
        next_step = "calculation_agent"
    else:
        next_step = "qa_agent"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent.model_dump(mode="json"),
        "next_step": next_step,
    }


def _run_task(state: AgentState, config: RunnableConfig, intent_type: str,
              response_schema: type[BaseModel]) -> Dict[str, Any]:
    """Run a specialist and append only this turn's messages to the graph."""
    configurable = config["configurable"]
    messages = get_chat_prompt_template(intent_type).invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
        "conversation_summary": state.get("conversation_summary", ""),
        "active_documents": state.get("active_documents", []),
    }).to_messages()
    result, tools_used = invoke_react_agent(
        response_schema, messages, configurable["llm"], configurable["tools"]
    )
    # Validate even if a model integration returns a dict instead of a model.
    structured = response_schema.model_validate(result["structured_response"])
    # Generation time is application metadata, not a date inferred by the model.
    structured.timestamp = datetime.now(timezone.utc)
    payload = structured.model_dump(mode="json")
    # The ReAct result includes its input messages. Keep the new human message
    # and generated tool/assistant messages, excluding temporary system prompts.
    turn_messages = [messages[-1], *result["messages"][len(messages):]]
    turn_messages = [message for message in turn_messages if not isinstance(message, SystemMessage)]
    return {
        "messages": turn_messages,
        "current_response": payload,
        "tools_used": tools_used,
        "actions_taken": [f"{intent_type}_agent"],
        "next_step": "update_memory",
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    return _run_task(state, config, "qa", AnswerResponse)


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """

    return _run_task(state, config, "summarization", SummarizationResponse)

def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    return _run_task(state, config, "calculation", CalculationResponse)


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        SystemMessagePromptTemplate.from_template(
            "Previous summary: {summary}\nPreviously active documents: {documents}"
        ),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
        "summary": state.get("conversation_summary", ""),
        "documents": state.get("active_documents", []),
    })

    structured_llm = llm.with_structured_output(
        UpdateMemoryResponse
    )

    response = structured_llm.invoke(prompt_with_history)
    return {
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=InMemorySaver())
