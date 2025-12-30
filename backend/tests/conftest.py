import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


# ============================================================================
# Vector Store Fixtures
# ============================================================================

@pytest.fixture
def mock_search_results():
    """Create sample SearchResults for testing"""
    return SearchResults(
        documents=[
            "MCP (Model Context Protocol) is a protocol for connecting AI models to data sources.",
            "MCP servers provide tools and resources to Claude through a standardized interface.",
        ],
        metadata=[
            {
                "course_title": "Introduction to MCP Servers",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Introduction to MCP Servers",
                "lesson_number": 2,
                "chunk_index": 5
            }
        ],
        distances=[0.15, 0.22],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Create error SearchResults"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: Connection timeout"
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore"""
    mock_store = Mock()
    mock_store.search = Mock()
    mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
    mock_store._resolve_course_name = Mock(return_value="Introduction to MCP Servers")
    return mock_store


# ============================================================================
# Course Data Fixtures
# ============================================================================

@pytest.fixture
def sample_course():
    """Create a sample Course object"""
    return Course(
        title="Introduction to MCP Servers",
        course_link="https://example.com/mcp-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is MCP?", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Building MCP Servers", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Advanced Patterns", lesson_link="https://example.com/lesson3"),
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample CourseChunk objects"""
    return [
        CourseChunk(
            content="MCP (Model Context Protocol) is a protocol for connecting AI models.",
            course_title="Introduction to MCP Servers",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="MCP servers provide tools and resources to Claude.",
            course_title="Introduction to MCP Servers",
            lesson_number=1,
            chunk_index=1
        ),
    ]


# ============================================================================
# Anthropic API Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()
    mock_client.messages = Mock()
    mock_client.messages.create = Mock()
    return mock_client


@pytest.fixture
def anthropic_direct_response():
    """Mock Anthropic response without tool use (direct answer)"""
    response = Mock()
    response.stop_reason = "end_turn"

    content_block = Mock()
    content_block.type = "text"
    content_block.text = "MCP stands for Model Context Protocol, a standard for connecting AI models to data sources."

    response.content = [content_block]
    return response


@pytest.fixture
def anthropic_tool_use_response():
    """Mock Anthropic response with tool_use (requests search)"""
    response = Mock()
    response.stop_reason = "tool_use"

    # Tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_01A2B3C4D5E6F7G8H9I0J1K2"
    tool_block.name = "search_course_content"
    tool_block.input = {
        "query": "What is MCP?",
        "course_name": "MCP"
    }

    response.content = [tool_block]
    return response


@pytest.fixture
def anthropic_final_response():
    """Mock Anthropic final response after tool execution"""
    response = Mock()
    response.stop_reason = "end_turn"

    content_block = Mock()
    content_block.type = "text"
    content_block.text = "MCP (Model Context Protocol) is a protocol for connecting AI models to data sources. It provides a standardized way for Claude to interact with external tools and resources through MCP servers."

    response.content = [content_block]
    return response


@pytest.fixture
def anthropic_tool_use_response_round2():
    """Mock Anthropic response with second tool_use (for sequential tool calling)"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_02B3C4D5E6F7G8H9I0J1K2L3"
    tool_block.name = "get_course_outline"
    tool_block.input = {"course_name": "MCP"}

    response.content = [tool_block]
    return response


@pytest.fixture
def anthropic_final_response_after_two_rounds():
    """Mock Anthropic final response after two rounds of tool execution"""
    response = Mock()
    response.stop_reason = "end_turn"

    content_block = Mock()
    content_block.type = "text"
    content_block.text = "Based on the course materials and outline, MCP consists of 3 lessons covering protocol basics, server implementation, and advanced patterns."

    response.content = [content_block]
    return response


# ============================================================================
# Tool Manager Fixtures
# ============================================================================

@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions = Mock(return_value=[
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ])
    mock_manager.execute_tool = Mock(return_value="[Introduction to MCP Servers - Lesson 1]\nMCP is a protocol for AI.")
    mock_manager.get_last_sources = Mock(return_value=[
        {"text": "Introduction to MCP Servers - Lesson 1", "link": "https://example.com/lesson1"}
    ])
    mock_manager.reset_sources = Mock()
    return mock_manager


# ============================================================================
# Session Manager Fixtures
# ============================================================================

@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager"""
    mock_manager = Mock()
    mock_manager.get_conversation_history = Mock(return_value=None)
    mock_manager.add_exchange = Mock()
    return mock_manager


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration"""
    from config import Config
    config = Config()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    return config


# ============================================================================
# FastAPI Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.query = Mock(return_value=(
        "This is a test response about MCP.",
        [{"text": "Test Course - Lesson 1", "link": "https://example.com/lesson1"}]
    ))
    mock_rag.get_course_analytics = Mock(return_value={
        "total_courses": 2,
        "course_titles": ["Introduction to MCP", "Advanced MCP"]
    })
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session = Mock(return_value="test_session_123")
    mock_rag.session_manager.delete_session = Mock(return_value=True)
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.
    This avoids issues with missing frontend directory during tests.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints (same as production but using mock_rag_system)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            deleted = mock_rag_system.session_manager.delete_session(session_id)
            return {
                "success": deleted,
                "message": "Session deleted" if deleted else "Session not found"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)
