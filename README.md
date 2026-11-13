# Document Assistant

A LangChain and LangGraph assistant that answers document questions, summarizes records, and calculates values using five sample financial and healthcare documents. The application lives in `starter/`.

## Run locally

Use Python 3.12 or later. From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r starter/requirements.txt
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY.
.venv/Scripts/python.exe starter/main.py
```

The default model is `gpt-4o` through the course endpoint `https://openai.vocareum.com/v1`. For a direct OpenAI API key, set `OPENAI_BASE_URL=https://api.openai.com/v1` in `.env`. The environment file is ignored by Git. Running the live assistant makes billed model requests.

Use `/docs` to list documents, `/help` for examples, and `/quit` to exit. The printed session ID can be resumed after restarting:

```powershell
.venv/Scripts/python.exe starter/main.py --session YOUR_SESSION_ID
```

Always run these commands from the repository root so `logs/` and `sessions/` are created there. Use the `starter/main.py` entry point; the root package scaffold is not the assistant implementation.

## Workflow and implementation decisions

`create_workflow()` builds a `StateGraph`, adds all five nodes, sets the classification entry point, configures conditional edges, and compiles with `InMemorySaver`:

```text
classify_intent -> qa_agent ------------> update_memory -> END
               -> summarization_agent -> update_memory -> END
               -> calculation_agent ---> update_memory -> END
```

Classification uses `UserIntent`, with `qa`, `summarization`, `calculation`, and `unknown` options. Unknown requests fall back to the Q&A node, which can ask for clarification. Prompt examples distinguish reading a stated amount from doing arithmetic. Specialists use intent-specific system prompts, prior messages, a conversation summary, and active document IDs.

Each specialist runs a ReAct tool loop using `document_search`, `document_reader`, `document_statistics`, and `calculator`. The retriever uses deterministic local keyword, document type, and amount filters; no embeddings or external database are required. Search supports all-document retrieval and exact document IDs.

The calculator returns numeric strings such as `"5"`. A character allowlist and bounded arithmetic AST permit real-number arithmetic while rejecting Python calls, attributes, complex values, oversized expressions, and excessive exponents. Every success or error is logged. Numeric results are finite and limited to a magnitude of `1e100`; expressions are limited to 256 characters and 64 AST nodes, with exponent magnitude at most 100. `%` means remainder; write `value * 0.10` for ten percent. Python floating-point behavior applies.

## Structured output

`AnswerResponse` requires `question`, `answer`, and confidence between 0 and 1, and supplies an independent empty sources list and creation timestamp by default. `UserIntent` requires a valid literal intent, confidence between 0 and 1, and string reasoning. Pydantic rejects invalid field types and out-of-range confidence. Its normal numeric coercion remains enabled.

Intent classification and memory updates use `with_structured_output()`. Specialist ReAct agents use their corresponding response schema (`AnswerResponse`, `SummarizationResponse`, or `CalculationResponse`). The specialist validates the returned model or dictionary again before placing JSON-compatible data in state. Response timestamps are assigned by the application in UTC, overriding any date supplied by the model. `process_message()` returns that data as `structured_response` and displays its answer, summary, or explanation. It does not display the separate unvalidated draft response.

## State and memory

The graph holds user input, messages, intent, next step, summary, active document IDs, validated current response, tools used, session/user IDs, and actions taken. `add_messages` merges conversation messages. Nodes append their names through the `operator.add` action reducer.

At each turn, `Overwrite` restores the session's saved messages and clears the actions list, so actions describe only the current turn. Specialists append only the new turn's messages; temporary system prompts and old messages are not appended again. Tool reporting also includes only the current turn. See the official [Overwrite reference](https://reference.langchain.com/python/langgraph/types/Overwrite).

`InMemorySaver` keeps graph checkpoints under a thread ID equal to the session ID. Those checkpoints are process-local. Durable memory comes from `sessions/<session_id>.json`, which stores serialized LangChain messages, conversation summary, document context, timestamps, and compact per-turn records. A new assistant loads that file, reconstructs messages, and restores the graph on its next invocation. Legacy session files containing cumulative graph states are migrated when loaded. Session files are written by atomic replacement.

`logs/session_<session_id>.json` is created when a session starts and records tool name, inputs, output, timestamp, and session ID. Resuming loads previous entries and appends new ones. Each session uses its own logger and tools. Both directories are generated automatically, even before the first message.

Storage is intended for one local process; concurrent writes to the same session are not coordinated. Full message history is retained, so long conversations consume more context. Sample records are simulated and older financial totals reflect the metadata actually present in those records.

## Example conversations

These are illustrative expected interactions; wording and confidence vary with the model. The live demo below produces actual execution evidence.

**Q&A**

> User: Who is the client on contract CON-001?
>
> Assistant: Healthcare Partners LLC (CON-001).
>
> Route: classify_intent -> qa_agent -> update_memory.
>
> Tools: document_search and document_reader.

**Summarization**

> User: Summarize contract CON-001, including services and termination terms.
>
> Assistant: CON-001 provides platform access, 24/7 support, monthly analytics, and compliance monitoring over 12 months. The monthly fee is $15,000 and either party can terminate with 60 days written notice.
>
> Route: classify_intent -> summarization_agent -> update_memory.

**Calculation**

> User: Calculate the monthly fee from CON-001's total value and duration.
>
> Calculator input: 180000 / 12
>
> Calculator output: "15000.0"
>
> Assistant: $180,000 / 12 months = $15,000 per month (CON-001).
>
> Route: classify_intent -> calculation_agent -> update_memory.

**Memory and session resumption**

> User (after resuming the printed session ID): What were its termination terms again?
>
> Assistant: Either party may terminate CON-001 with 60 days written notice.

The saved history and summary identify the contract referred to by "its". `/docs` lists all five available records. Invalid calculator input or division by zero returns a logged error message. Model or persistence errors return `success: false` instead of claiming a completed response.

## Verification

Run offline regression tests without API calls:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The tests execute the real outer graph, ReAct tool loop, local tools, and persistence with a scripted chat model. They cover all three intents and unknown fallback, validation, calculator errors and safety, all-document and ID search, prompt selection, session resumption, legacy format migration, per-turn actions and tools, history deduplication, log preservation, isolation, and model failure. They verify application integration; the scripted model does not measure real LLM classification quality.

To create real model evidence for all three intents and a resumed follow-up:

```powershell
.venv/Scripts/python.exe scripts/demo.py
```

This requires a valid configured API key and network access. It prints validated responses, tool usage, intent, sources, actions, summary, and the generated session/log filenames. Check those artifacts when reviewing the rubric. The implementation retains the project's `create_react_agent` API; LangGraph 1.x marks it deprecated, so migrating the tool loop is needed before upgrading to LangGraph 2.x.

### Verified execution evidence

All 10 offline regression tests pass. A completed live demo classified Q&A, summarization, and calculation correctly, used the calculator for `180000 / 12 = 15000`, and answered a contract termination follow-up after a new assistant instance resumed the session. Generated response timestamps were checked against application time; saved messages contain exactly four user turns.

- [Live conversation session](sessions/731d05b8-6c5d-4ba6-a6e4-d1979ff34714.json)
- [Session tool calls](logs/session_731d05b8-6c5d-4ba6-a6e4-d1979ff34714.json)
- [Live demo transcript](logs/demo_execution.txt)
