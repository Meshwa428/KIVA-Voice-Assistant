
### 🏛️ High-Level Architecture

We will implement a **Pipeline Architecture** using Python's `asyncio` and `Queues`.

1.  **Input Layer:** `RealtimeSTT` runs in a thread, pushing audio/text chunks to an `InputQueue`.
2.  **Orchestrator (The Brain):** An async consumer reads the `InputQueue`. It decides:
    *   Is this a chat? -> Stream to LLM.
    *   Is this a command? -> Route to the appropriate **Agent**.
3.  **Agent/Tool Layer:**
    *   Agents are specialized (e.g., `CoderAgent`, `SearchAgent`).
    *   Tools are provided via **MCP (Model Context Protocol)** clients.
    *   The LLM generates "Tool Calls" which the Agent executes, then feeds results back to the LLM.
4.  **Output Layer (Streaming):**
    *   The LLM text stream is pushed to a `TextStreamQueue`.
    *   **Optimization:** We will subclass `RealtimeTTS.BaseEngine` to wrap your **Supertonic** model. `RealtimeTTS` *already* handles the sentence buffering (detecting full sentences from a stream) and async playback. We do not need a separate sentence transformer for this (it is too slow for real-time). Simple linguistic heuristics (punctuation lookahead) used by RealtimeTTS are much faster.

---

### 📂 File Structure

```text
kiva/
├── .env                    # Secrets and configuration
├── config.yaml             # Model paths, voice settings, MCP server URLs
├── pyproject.toml          # Dependencies (managed by uv)
├── assets/
│   ├── onnx/               # Supertonic models
│   └── voice_styles/       # Voice JSONs
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point (Async loop starter)
│   │
│   ├── core/               # Core infrastructure
│   │   ├── __init__.py
│   │   ├── config.py       # Config loader
│   │   ├── events.py       # Event bus (for Agent-to-Agent comms)
│   │   └── logger.py       # Centralized logging
│   │
│   ├── audio/              # Audio Input/Output
│   │   ├── __init__.py
│   │   ├── stt_service.py  # Wrapper around RealtimeSTT
│   │   ├── tts_engine.py   # Custom RealtimeTTS Engine for Supertonic (Crucial)
│   │   └── player.py       # PyAudio stream handler
│   │
│   ├── llm/                # Brains
│   │   ├── __init__.py
│   │   ├── client.py       # Async Ollama client
│   │   ├── prompts.py      # System prompts (Personality, Tool instructions)
│   │   └── stream_parser.py# Logic to detect tool calls vs speech in stream
│   │
│   ├── agents/             # The "Workers"
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract Agent class
│   │   ├── orchestrator.py # The Router (Classifies intent)
│   │   ├── chat.py         # Standard Conversational Agent
│   │   └── task.py         # Agent capable of executing complex workflows
│   │
│   └── tools/              # MCP Implementation
│       ├── __init__.py
│       ├── registry.py     # Tool Registry (Lookup)
│       └── mcp_client.py   # Client to talk to MCP Servers (Filesystem, Brave, etc.)
│
└── tests/                  # Unit tests
```

---

### 🔨 Key Technical Implementation Details

#### 1. The Custom TTS Engine (`src/audio/tts_engine.py`)
Instead of manually handling buffering, we adapt Supertonic to the `RealtimeTTS` interface. This library is optimized for the exact latency requirement you have.

*We will create a class `SupertonicEngine` that inherits from `RealtimeTTS.engines.BaseEngine`. This allows us to plug your ONNX model directly into a proven streaming pipeline.*

#### 2. The MCP Client (`src/tools/mcp_client.py`)
This is the modern way to handle tools. Instead of hardcoding "get_weather", we connect to MCP Servers.
*   **Concept:** The Assistant acts as an *MCP Client*.
*   **Action:** It connects to local or remote MCP servers (e.g., a standardized filesystem server).
*   **Discovery:** On startup, the registry queries connected MCP servers for their tool definitions and injects them into the LLM system prompt.

#### 3. Sentence Detection Optimization (Fact Check)
> *Your suggestion:* "use sentence transformer to detect if we have 1 sentence"

**Improvement:** Do **not** use a Sentence Transformer for boundary detection in a low-latency loop.
*   **Why?** Encoding every chunk of text through a transformer adds significant inference latency (50ms–200ms per chunk).
*   **Better Approach:** Use **Token Lookahead + Heuristics**. The `RealtimeTTS` library (which we will utilize) uses a fast "stream-to-sentence" buffer that looks for punctuation (`.`, `?`, `!`) and pauses. This takes <1ms and feels instant to the user.

#### 4. Asyncio Orchestration
We will not use a `while True` loop. We will use an `asyncio.Event` loop.

```python
# Pseudo-code for src/agents/orchestrator.py
async def run(self):
    while True:
        # 1. Await Audio Input
        user_text = await input_queue.get()
        
        # 2. Add to context
        self.memory.add(user_text)
        
        # 3. Stream from LLM
        async for chunk in self.llm.stream(context):
            # 4. Check if chunk contains a tool call tag (e.g., <tool>...</tool>)
            if self.detector.is_tool_call(chunk):
                # Pause TTS, execute tool, feed result back to LLM
                await self.execute_tool(chunk)
            else:
                # 5. Push text to TTS Stream immediately
                self.tts_stream.feed(chunk)
```

---

### 📅 Project Plan

#### Phase 1: The Core & Audio Engine (Day 1-2)
*   **Goal:** Replicate your current script but in the modular structure using `RealtimeTTS` architecture.
*   **Tasks:**
    *   [x] Set up `uv` project structure.
    *   [x] Create `src/audio/tts_engine.py`: Wrap the Supertonic ONNX code into a class that accepts a stream of text and yields PCM audio bytes.
    *   [x] Implement Custom Wakeword Engine (`src/audio/wakeword.py`) to decouple from RealtimeSTT's internal limitation.
    *   [x] Integrate `RealtimeSTT` (moved to `src/audio/stt_service.py`).
    *   [x] Get basic "Echo" functionality working (Speak -> STT -> TTS).

#### Phase 2: The Async Brain (Day 3)
*   **Goal:** Connect the LLM asynchronously.
*   **Tasks:**
    *   [x] Implement `src/llm/client.py` using `ollama` (or an async wrapper like `aiohttp` against Ollama API).
    *   [x] Create the "sentence buffer" pipeline (LLM stream -> Buffer -> TTS Engine).
    *   [ ] Verify latency is under 500ms.

#### Phase 3: MCP & Tool Registry (Day 4-5)
*   **Goal:** Give Kiva hands.
*   **Tasks:**
    *   Implement a basic MCP Client (connecting to a simple stdio MCP server, e.g., a file reader).
    *   Create `src/tools/registry.py` to format tools into JSON schemas for the LLM.
    *   Update System Prompt to understand how to call tools.

#### Phase 4: Agentic Routing (Day 6)
*   **Goal:** Autonomy.
*   **Tasks:**
    *   Implement the Orchestrator.
    *   Handle the loop: `LLM -> Tool Call -> Execute -> Result -> LLM -> Final Answer`.
    *   Handle "Interruption" (if user speaks while Kiva is talking/acting, stop the specific task).