import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

# ============================================================================
# CourseSearchTool Tests
# ============================================================================


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        # Verify structure
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Verify schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_successful_search(self, mock_vector_store, mock_search_results):
        """Test successful search returns formatted results"""
        # Setup
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        # Execute
        result = tool.execute(query="What is MCP?")

        # Verify
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

        assert isinstance(result, str)
        assert "[Introduction to MCP Servers - Lesson 1]" in result
        assert "MCP (Model Context Protocol)" in result
        assert len(tool.last_sources) == 2

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test search with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is MCP?", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="MCP", lesson_number=None
        )
        assert "Introduction to MCP Servers" in result

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is MCP?", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=1
        )
        assert "Lesson 1" in result

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test search with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(
            query="What is MCP?", course_name="MCP Servers", lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="MCP Servers", lesson_number=1
        )

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = empty_search_results

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result
        assert tool.last_sources == []

    def test_execute_empty_results_with_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes filter information"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = empty_search_results

        result = tool.execute(
            query="test", course_name="Advanced Course", lesson_number=5
        )

        assert "No relevant content found" in result
        assert "in course 'Advanced Course'" in result
        assert "in lesson 5" in result

    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test that search errors are returned as strings"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = error_search_results

        result = tool.execute(query="test")

        assert "Search error: Connection timeout" in result
        assert isinstance(result, str)

    def test_format_results_with_lesson_links(
        self, mock_vector_store, mock_search_results
    ):
        """Test that lesson links are retrieved and stored in sources"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = tool.execute(query="test")

        # Verify lesson link retrieval was called
        assert mock_vector_store.get_lesson_link.call_count == 2

        # Verify sources contain links
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
        assert tool.last_sources[0]["text"] == "Introduction to MCP Servers - Lesson 1"

    def test_format_results_missing_lesson_number(self, mock_vector_store):
        """Test formatting when metadata is missing lesson_number"""
        tool = CourseSearchTool(mock_vector_store)

        # Create results without lesson numbers
        results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = results

        result = tool.execute(query="test")

        assert "[Test Course]" in result
        assert "Lesson" not in result  # Should not include lesson number
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None

    def test_source_tracking_resets_between_searches(
        self, mock_vector_store, mock_search_results
    ):
        """Test that last_sources is properly updated on each search"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        # First search
        tool.execute(query="first query")
        first_sources = tool.last_sources.copy()
        assert len(first_sources) == 2

        # Second search with different results
        single_result = SearchResults(
            documents=["Single result"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = single_result

        tool.execute(query="second query")
        assert len(tool.last_sources) == 1
        assert tool.last_sources != first_sources


# ============================================================================
# ToolManager Tests
# ============================================================================


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_register_tool_without_name_raises_error(self):
        """Test that registering tool without name raises ValueError"""
        manager = ToolManager()

        # Create mock tool with invalid definition
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(bad_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test retrieving all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store, mock_search_results):
        """Test executing a registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        mock_vector_store.search.return_value = mock_search_results

        result = manager.execute_tool(
            "search_course_content", query="test", course_name="MCP"
        )

        assert isinstance(result, str)
        assert "MCP" in result

    def test_execute_nonexistent_tool(self):
        """Test executing tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, mock_search_results):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        mock_vector_store.search.return_value = mock_search_results

        # Execute search
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]["text"] == "Introduction to MCP Servers - Lesson 1"

    def test_reset_sources(self, mock_vector_store, mock_search_results):
        """Test resetting sources across all tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        mock_vector_store.search.return_value = mock_search_results

        # Execute search and verify sources exist
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) == 2

        # Reset sources
        manager.reset_sources()

        assert manager.get_last_sources() == []
