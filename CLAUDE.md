# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using ChromaDB for vector search and Anthropic's Claude API with tool-calling capabilities.

## Important: Server Management

**NEVER automatically run the development server using `./run.sh` or `uv run uvicorn` commands.** The user will start the server manually themselves. Do not use Bash commands to start, stop, or restart the server unless explicitly requested by the user.

## Commands

### Development

**CRITICAL: This project uses `uv` for ALL dependency management and command execution.**

**Do NOT use:**
- `pip install` / `pip freeze` / `pip uninstall`
- `python -m venv`
- `python script.py` - Always use `uv run python script.py`
- `uvicorn app:app` - Always use `uv run uvicorn app:app`
- Direct execution of any Python files without `uv run`

**Always use `uv` commands:**

```bash
# Install/sync dependencies from pyproject.toml and uv.lock
uv sync

# Add a new dependency (updates pyproject.toml and uv.lock)
uv add <package-name>

# Remove a dependency
uv remove <package-name>

# Run the application (recommended)
./run.sh

# Run manually - ALWAYS use "uv run"
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any Python script or command
uv run python script.py
uv run pytest
uv run <any-command>

# Access the application
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Code Quality Tools

This project uses automated code quality tools to maintain consistency:

**Tools:**
- **Black**: Automatic code formatting (line length: 88)
- **isort**: Import sorting (compatible with Black)
- **flake8**: Linting and style checking

**Scripts:**

```bash
# Auto-format all Python code (runs black and isort)
./format.sh

# Check code quality without making changes
./quality-check.sh

# Run individual tools
uv run black backend/ main.py
uv run isort backend/ main.py
uv run flake8 backend/ main.py
```

**Configuration:**
- Tool settings are in `pyproject.toml` ([tool.black], [tool.isort])
- Flake8 config in `.flake8` file
- Run `./format.sh` before committing to ensure code consistency

### Environment Setup

Required environment variable in `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

### Core Data Flow

1. **User Query** → Frontend (`script.js`) sends POST to `/api/query`
2. **RAG System** → Orchestrates query processing through components
3. **Claude Tool-Calling** → Claude decides whether to search course content
4. **Vector Search** → ChromaDB searches embeddings (if Claude invokes tool)
5. **Response Generation** → Claude synthesizes answer from retrieved context
6. **Session Management** → Conversation history stored for context

### Component Responsibilities

**backend/rag_system.py** (Orchestrator)
- Coordinates all RAG components
- Entry point: `query(query, session_id)` method
- Manages document loading from `/docs` folder
- Handles tool registration and execution flow

**backend/ai_generator.py** (Claude Integration)
- System prompt enforces "one search per query" rule and concise responses
- Tool execution via multi-turn conversation with Claude API
- `_handle_tool_execution()` manages tool call → result → final response flow
- Temperature: 0, Max tokens: 800

**backend/search_tools.py** (Tool System)
- Abstract `Tool` class for extensibility
- `CourseSearchTool` implements search with semantic course name matching
- `ToolManager` handles tool registration and execution routing
- Tracks sources from searches for UI attribution

**backend/vector_store.py** (ChromaDB Interface)
- Two collections:
  - `course_catalog` - Course metadata for semantic course lookup
  - `course_content` - Chunked lesson content with metadata
- Unified `search()` method handles course name resolution and filtering
- Embedding model: `all-MiniLM-L6-v2` (Sentence Transformers)

**backend/document_processor.py** (Document Pipeline)
- Parses structured course format:
  - Line 1: `Course Title: [title]`
  - Line 2: `Course Link: [url]`
  - Line 3: `Course Instructor: [name]`
  - Following: `Lesson N: [title]` with content
- Sentence-based chunking (800 chars, 100 char overlap)
- Enriches chunks with context: "Course {title} Lesson {N} content: ..."

**backend/session_manager.py** (Conversation State)
- Session-based conversation history (MAX_HISTORY: 2 messages)
- Provides formatted history to Claude for context continuity

### Key Configuration (backend/config.py)

```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_RESULTS = 5
MAX_HISTORY = 2
CHROMA_PATH = "./chroma_db"
```

## Important Implementation Details

### Tool-Based RAG Pattern

This codebase uses **tool-calling RAG**, not traditional "retrieve-then-generate":
- Claude receives tool definitions on each query
- Claude autonomously decides when to search based on query type
- General knowledge questions: Claude answers directly without searching
- Course-specific questions: Claude invokes `search_course_content` tool first

**System prompt constraints** (in ai_generator.py):
- One search per query maximum (prevents search loops)
- No meta-commentary about search process
- Direct, concise educational responses

### Course Name Resolution

`VectorStore.search()` handles fuzzy course matching:
1. If `course_name` provided, searches `course_catalog` collection semantically
2. Gets actual course title from top match
3. Filters `course_content` search by resolved course title
4. Enables queries like "MCP" to match "Introduction to MCP Servers"

### Document Loading Strategy

`RAGSystem.add_course_folder()`:
- Checks existing course titles to avoid duplicates
- Uses course title as unique identifier
- Set `clear_existing=True` to rebuild from scratch
- Called on startup via `app.py` startup event

### Chunk Context Enrichment

Each chunk includes contextual prefix for better retrieval:
```python
# First chunk of lesson:
"Lesson {N} content: {chunk}"

# Last lesson chunks:
"Course {title} Lesson {N} content: {chunk}"
```

This helps Claude understand source context when synthesizing answers.

### Frontend Integration

**frontend/script.js**:
- Generates unique session IDs for conversation tracking
- Renders markdown responses using Marked.js
- Displays source attributions (course + lesson) below answers
- Polls `/api/courses` for sidebar statistics

## Common Workflows

### Adding New Tools

1. Create tool class implementing `Tool` abstract class in `search_tools.py`
2. Implement `get_tool_definition()` (Anthropic tool schema)
3. Implement `execute(**kwargs)` method
4. Register in `RAGSystem.__init__()`: `self.tool_manager.register_tool(YourTool())`

### Modifying Search Behavior

Primary file: `backend/vector_store.py`
- Adjust `MAX_RESULTS` in config.py for more/fewer results
- Modify distance thresholds in `_filter_results_by_distance()`
- Change similarity metric in ChromaDB collection creation

### Changing Chunking Strategy

Primary file: `backend/document_processor.py`
- `chunk_text()` uses sentence-based splitting
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in config.py
- Current regex handles abbreviations (e.g., "Dr.", "U.S.")

### Adjusting Response Style

Primary file: `backend/ai_generator.py`
- Modify `SYSTEM_PROMPT` static variable
- Adjust `max_tokens` in `base_params` (currently 800)
- Change `temperature` for more/less deterministic responses (currently 0)

## Data Models (backend/models.py)

```python
Course(title, course_link, instructor, lessons: List[Lesson])
Lesson(lesson_number, title, lesson_link)
CourseChunk(content, course_title, lesson_number, chunk_index)
Message(role, content)  # For conversation history
SearchResults(documents, metadata, distances, error)
```

Course title serves as the unique identifier throughout the system.

## Startup Behavior

When server starts (`app.py` startup event):
1. Checks if `../docs` folder exists
2. Calls `rag_system.add_course_folder(docs_path, clear_existing=False)`
3. Skips already-loaded courses (deduplication by title)
4. Prints course count and chunk count to console

This means ChromaDB persists between restarts. Delete `./chroma_db` folder to force fresh indexing.

## File Organization Principles

- **backend/app.py** - API routes only, minimal logic
- **backend/rag_system.py** - Orchestration, no direct API/storage/AI calls
- Component files (vector_store, ai_generator, etc.) - Single responsibility
- **frontend/** - Static files, no build step required
- **docs/** - Course materials in structured text format

## Testing Considerations

Currently no test suite. When adding tests:
- Mock Anthropic API calls in `ai_generator.py`
- Use in-memory ChromaDB for vector store tests
- Test document parser with sample course files
- Test session manager conversation history limits
