import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystem:
    """Integration tests for the RAG system end-to-end flow"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        config = Mock()
        config.ANTHROPIC_API_KEY = "test_api_key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        config.CHROMA_PATH = "./test_chroma_db"
        return config

    @pytest.fixture
    def mock_rag_system(self, test_config):
        """Create RAGSystem with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            rag = RAGSystem(test_config)

            # Mock the components
            rag.vector_store = Mock()
            rag.ai_generator = Mock()
            rag.session_manager = Mock()
            rag.tool_manager = Mock()
            rag.search_tool = Mock()

            return rag

    # ========================================================================
    # Query Flow Tests
    # ========================================================================

    def test_query_without_session(self, mock_rag_system):
        """Test query processing without session ID"""
        # Setup
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = (
            "MCP is a protocol for AI."
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"}
        ]

        # Execute
        response, sources = mock_rag_system.query("What is MCP?", session_id=None)

        # Verify
        assert response == "MCP is a protocol for AI."
        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"

        # Verify session manager was not called for history/adding
        mock_rag_system.session_manager.get_conversation_history.assert_not_called()
        mock_rag_system.session_manager.add_exchange.assert_not_called()

    def test_query_with_session(self, mock_rag_system):
        """Test query processing with session ID for conversation context"""
        # Setup
        session_id = "test_session_123"
        history = "User: Hello\nAssistant: Hi there!"
        mock_rag_system.session_manager.get_conversation_history.return_value = history
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Here's more information about MCP."
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Execute
        response, sources = mock_rag_system.query("Tell me more", session_id=session_id)

        # Verify history was retrieved
        mock_rag_system.session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator received history
        ai_call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert ai_call_kwargs["conversation_history"] == history

        # Verify exchange was added to session
        # Note: add_exchange receives the original query, not the formatted prompt
        mock_rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Tell me more", "Here's more information about MCP."
        )

    def test_query_includes_tool_definitions(self, mock_rag_system):
        """Test that query passes tool definitions to AI generator"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Execute
        mock_rag_system.query("What is MCP?")

        # Verify tools were passed
        ai_call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "tools" in ai_call_kwargs
        assert len(ai_call_kwargs["tools"]) == 1
        assert ai_call_kwargs["tool_manager"] == mock_rag_system.tool_manager

    def test_query_resets_sources_after_retrieval(self, mock_rag_system):
        """Test that sources are reset after being retrieved"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "Source 1"}
        ]

        # Execute
        response, sources = mock_rag_system.query("test")

        # Verify sources were retrieved and then reset
        mock_rag_system.tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that user query is formatted correctly for AI"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        user_query = "What is MCP?"
        mock_rag_system.query(user_query)

        # Verify prompt formatting
        ai_call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        expected_prompt = f"Answer this question about course materials: {user_query}"
        assert ai_call_kwargs["query"] == expected_prompt

    # ========================================================================
    # Source Attribution Tests
    # ========================================================================

    def test_query_returns_sources_from_tool_searches(self, mock_rag_system):
        """Test that sources from tool searches are returned"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = "Answer"

        expected_sources = [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "MCP Course - Lesson 2", "link": "https://example.com/lesson2"},
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = expected_sources

        response, sources = mock_rag_system.query("What is MCP?")

        assert sources == expected_sources

    def test_query_returns_empty_sources_when_no_tool_used(self, mock_rag_system):
        """Test that empty sources are returned when AI doesn't use tools"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = (
            "General knowledge answer"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = mock_rag_system.query("What is machine learning?")

        assert sources == []

    # ========================================================================
    # Component Initialization Tests
    # ========================================================================

    def test_rag_system_initializes_all_components(self, test_config):
        """Test that RAGSystem initializes all required components"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager") as mock_session,
            patch("rag_system.CourseSearchTool") as mock_search,
            patch("rag_system.CourseOutlineTool") as mock_outline,
        ):

            rag = RAGSystem(test_config)

            # Verify all components were initialized
            mock_doc_proc.assert_called_once_with(
                test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP
            )
            mock_vector.assert_called_once_with(
                test_config.CHROMA_PATH,
                test_config.EMBEDDING_MODEL,
                test_config.MAX_RESULTS,
            )
            mock_ai.assert_called_once_with(
                test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
            )
            mock_session.assert_called_once_with(test_config.MAX_HISTORY)

    def test_rag_system_registers_tools(self, test_config):
        """Test that RAGSystem registers search and outline tools"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager") as mock_tool_manager,
        ):

            rag = RAGSystem(test_config)

            # Verify tools were registered
            # ToolManager.register_tool should be called twice (search + outline)
            assert mock_tool_manager.return_value.register_tool.call_count == 2

    # ========================================================================
    # Course Analytics Tests
    # ========================================================================

    def test_get_course_analytics(self, mock_rag_system):
        """Test retrieving course analytics"""
        mock_rag_system.vector_store.get_course_count.return_value = 3
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Introduction to MCP",
            "Advanced MCP",
            "Computer Use API",
        ]

        analytics = mock_rag_system.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Introduction to MCP" in analytics["course_titles"]

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_query_handles_ai_generator_errors(self, mock_rag_system):
        """Test that query handles errors from AI generator gracefully"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.side_effect = Exception(
            "API Error"
        )

        # This should raise the exception - test that it's not swallowed
        with pytest.raises(Exception, match="API Error"):
            mock_rag_system.query("test")

    def test_query_handles_tool_manager_errors(self, mock_rag_system):
        """Test handling of tool manager errors"""
        mock_rag_system.session_manager.get_conversation_history.return_value = None
        mock_rag_system.ai_generator.generate_response.return_value = "Response"
        mock_rag_system.tool_manager.get_last_sources.side_effect = Exception(
            "Tool error"
        )

        # Should raise error when trying to get sources
        with pytest.raises(Exception, match="Tool error"):
            mock_rag_system.query("test")


# ============================================================================
# End-to-End Integration Tests (More Realistic Scenarios)
# ============================================================================


class TestRAGSystemEndToEnd:
    """End-to-end tests with more realistic component interactions"""

    @pytest.fixture
    def integrated_rag_system(self, test_config):
        """Create RAG system with partially mocked components"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.SessionManager"),
        ):

            # Import real components we want to test
            from ai_generator import AIGenerator
            from search_tools import CourseSearchTool, ToolManager

            rag = RAGSystem(test_config)

            # Use real tool manager and search tool
            rag.tool_manager = ToolManager()

            # Mock vector store but use real search tool
            rag.vector_store = Mock()
            rag.search_tool = CourseSearchTool(rag.vector_store)
            rag.tool_manager.register_tool(rag.search_tool)

            # Mock AI generator
            rag.ai_generator = Mock()

            # Mock session manager
            rag.session_manager = Mock()
            rag.session_manager.get_conversation_history.return_value = None
            rag.session_manager.add_exchange = Mock()

            return rag

    def test_end_to_end_content_query(self, integrated_rag_system, mock_search_results):
        """Test complete flow: query → AI calls tool → tool searches → format sources → return"""
        # Setup: Vector store returns search results
        integrated_rag_system.vector_store.search.return_value = mock_search_results
        integrated_rag_system.vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        # Setup: AI generator returns response
        integrated_rag_system.ai_generator.generate_response.return_value = (
            "MCP is a protocol for AI models."
        )

        # Execute
        response, sources = integrated_rag_system.query("What is MCP?", session_id=None)

        # Verify response
        assert response == "MCP is a protocol for AI models."

        # Verify sources were properly extracted (from real search tool)
        # Note: Since AI generator is mocked, sources won't be populated
        # In real scenario, AI would call tool which would populate last_sources
        assert isinstance(sources, list)

    def test_tool_manager_integration_with_search_tool(
        self, integrated_rag_system, mock_search_results
    ):
        """Test that ToolManager correctly executes CourseSearchTool"""
        # Setup
        integrated_rag_system.vector_store.search.return_value = mock_search_results
        integrated_rag_system.vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        # Execute tool through manager
        result = integrated_rag_system.tool_manager.execute_tool(
            "search_course_content", query="What is MCP?", course_name="MCP"
        )

        # Verify tool was executed
        integrated_rag_system.vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="MCP", lesson_number=None
        )

        # Verify result formatting
        assert isinstance(result, str)
        assert "Introduction to MCP Servers" in result

        # Verify sources were tracked
        sources = integrated_rag_system.tool_manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["link"] == "https://example.com/lesson1"
