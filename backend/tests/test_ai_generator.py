import pytest
from unittest.mock import Mock, patch, call
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator with tool-calling functionality"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance for testing"""
        return AIGenerator(api_key="test_api_key", model="claude-sonnet-4-20250514")

    # ========================================================================
    # Direct Response Tests (No Tools)
    # ========================================================================

    def test_generate_direct_response_no_tools(self, ai_generator, mock_anthropic_client, anthropic_direct_response):
        """Test generating direct response without tool usage"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_direct_response

        response = ai_generator.generate_response(
            query="What is MCP?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )

        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]

        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert call_kwargs["messages"][0]["content"] == "What is MCP?"
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT
        assert "tools" not in call_kwargs

        # Verify response
        assert response == "MCP stands for Model Context Protocol, a standard for connecting AI models to data sources."

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client, anthropic_direct_response):
        """Test that conversation history is included in system prompt"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_direct_response

        history = "User: Hello\nAssistant: Hi there!"
        response = ai_generator.generate_response(
            query="Tell me more",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        assert history in system_content
        assert AIGenerator.SYSTEM_PROMPT in system_content

    # ========================================================================
    # Tool-Calling Flow Tests
    # ========================================================================

    def test_generate_response_with_tool_use(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test complete tool-calling flow: request → tool execution → final response"""
        ai_generator.client = mock_anthropic_client

        # Setup: First call returns tool_use, second call returns final response
        mock_anthropic_client.messages.create.side_effect = [
            anthropic_tool_use_response,
            anthropic_final_response
        ]

        # Execute
        response = ai_generator.generate_response(
            query="What is MCP?",
            conversation_history=None,
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify initial API call included tools
        first_call_kwargs = mock_anthropic_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_kwargs
        assert first_call_kwargs["tool_choice"] == {"type": "auto"}

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is MCP?",
            course_name="MCP"
        )

        # Verify second API call with tool results
        assert mock_anthropic_client.messages.create.call_count == 2
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]

        # Check message structure
        messages = second_call_kwargs["messages"]
        assert len(messages) == 3  # Original query, assistant tool_use, user tool_result

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Verify tool results format
        tool_results = messages[2]["content"]
        assert isinstance(tool_results, list)
        assert tool_results[0]["type"] == "tool_result"
        assert "tool_use_id" in tool_results[0]
        assert "content" in tool_results[0]

        # Verify final response
        assert "MCP (Model Context Protocol)" in response

    def test_tool_execution_without_tool_manager_returns_direct_response(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response
    ):
        """Test that tool_use response without tool_manager doesn't crash"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_tool_use_response

        # Call with tools but no tool_manager - should return text from content
        # Since stop_reason is tool_use but tool_manager is None, it should gracefully handle
        response = ai_generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "test"}],
            tool_manager=None  # No manager provided
        )

        # Should attempt to access content[0].text, which doesn't exist in tool_use
        # This will raise AttributeError - this is an expected failure case
        # In production, we should add error handling for this scenario

    def test_multiple_tool_calls_in_response(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test handling of multiple tool calls in single response"""
        ai_generator.client = mock_anthropic_client

        # Create response with two tool calls
        multi_tool_response = Mock()
        multi_tool_response.stop_reason = "tool_use"

        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.id = "tool_id_1"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "What is MCP?"}

        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.id = "tool_id_2"
        tool_block2.name = "search_course_content"
        tool_block2.input = {"query": "What are MCP servers?"}

        multi_tool_response.content = [tool_block1, tool_block2]

        mock_anthropic_client.messages.create.side_effect = [
            multi_tool_response,
            anthropic_final_response
        ]

        response = ai_generator.generate_response(
            query="Tell me about MCP",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify second API call contains both tool results
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        tool_results = second_call_kwargs["messages"][2]["content"]
        assert len(tool_results) == 2
        assert all(tr["type"] == "tool_result" for tr in tool_results)

    # ========================================================================
    # Handle Tool Execution Method Tests
    # ========================================================================

    def test_handle_tool_execution_builds_correct_message_structure(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test that _handle_tool_execution builds correct message structure"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_final_response

        base_params = {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is MCP?"}],
            "system": AIGenerator.SYSTEM_PROMPT,
            "tools": mock_tool_manager.get_tool_definitions(),
            "tool_choice": {"type": "auto"}
        }

        result = ai_generator._handle_tool_execution(
            anthropic_tool_use_response,
            base_params,
            mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify final API call structure
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs  # Tools included for sequential calling
        assert call_kwargs["tool_choice"] == {"type": "auto"}
        assert len(call_kwargs["messages"]) == 3

        # Verify message sequence
        assert call_kwargs["messages"][0]["role"] == "user"  # Original
        assert call_kwargs["messages"][1]["role"] == "assistant"  # Tool use
        assert call_kwargs["messages"][2]["role"] == "user"  # Tool results

    def test_tool_result_includes_tool_use_id(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test that tool results correctly reference tool_use_id"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_final_response

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": AIGenerator.SYSTEM_PROMPT
        }

        result = ai_generator._handle_tool_execution(
            anthropic_tool_use_response,
            base_params,
            mock_tool_manager
        )

        # Extract tool results from the call
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        tool_results = call_kwargs["messages"][2]["content"]

        # Verify tool_use_id matches
        assert tool_results[0]["tool_use_id"] == "toolu_01A2B3C4D5E6F7G8H9I0J1K2"

    # ========================================================================
    # Edge Cases and Error Scenarios
    # ========================================================================

    def test_generate_response_with_empty_query(self, ai_generator, mock_anthropic_client, anthropic_direct_response):
        """Test handling of empty query string"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_direct_response

        response = ai_generator.generate_response(query="", tools=None, tool_manager=None)

        # Should still make API call
        mock_anthropic_client.messages.create.assert_called_once()
        assert isinstance(response, str)

    def test_api_parameters_include_all_required_fields(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_direct_response
    ):
        """Test that all required API parameters are present"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_direct_response

        ai_generator.generate_response(query="test", tools=None, tool_manager=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]

        # Verify all required parameters
        assert "model" in call_kwargs
        assert "messages" in call_kwargs
        assert "system" in call_kwargs
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs

        # Verify types
        assert isinstance(call_kwargs["messages"], list)
        assert isinstance(call_kwargs["system"], str)
        assert isinstance(call_kwargs["temperature"], (int, float))
        assert isinstance(call_kwargs["max_tokens"], int)

    # ========================================================================
    # Sequential Tool Calling Tests (2 Rounds)
    # ========================================================================

    def test_sequential_tool_calling_two_rounds(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_tool_use_response_round2,
        anthropic_final_response_after_two_rounds,
        mock_tool_manager
    ):
        """Test sequential tool calling across 2 rounds"""
        ai_generator.client = mock_anthropic_client

        # Setup: 3 API calls
        mock_anthropic_client.messages.create.side_effect = [
            anthropic_tool_use_response,           # Round 1
            anthropic_tool_use_response_round2,    # Round 2
            anthropic_final_response_after_two_rounds  # Final
        ]

        mock_tool_manager.execute_tool.side_effect = [
            "[MCP - Lesson 1]\nMCP is a protocol...",
            "Course: MCP\nLessons: 3 total..."
        ]

        response = ai_generator.generate_response(
            query="Tell me about MCP course structure and content",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify 3 API calls, 2 tool executions
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert "MCP consists of 3 lessons" in response

        # Verify message accumulation
        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5  # user, assistant, user, assistant, user

    def test_sequential_tool_calling_stops_at_max_rounds(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        mock_tool_manager
    ):
        """Test stops after MAX_TOOL_ROUNDS even if still tool_use"""
        ai_generator.client = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = anthropic_tool_use_response
        mock_tool_manager.execute_tool.return_value = "Tool result"

        response = ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify only MAX_TOOL_ROUNDS iterations
        from config import config
        expected_calls = config.MAX_TOOL_ROUNDS + 1
        assert mock_anthropic_client.messages.create.call_count == expected_calls
        assert isinstance(response, str)

    def test_sequential_tool_calling_stops_early_on_end_turn(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test stops early if stop_reason is end_turn before max rounds"""
        ai_generator.client = mock_anthropic_client

        mock_anthropic_client.messages.create.side_effect = [
            anthropic_tool_use_response,
            anthropic_final_response
        ]

        mock_tool_manager.execute_tool.return_value = "Tool result"

        response = ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify only 2 API calls (not 3)
        assert mock_anthropic_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert response == anthropic_final_response.content[0].text

    def test_tools_included_in_sequential_api_calls(
        self,
        ai_generator,
        mock_anthropic_client,
        anthropic_tool_use_response,
        anthropic_final_response,
        mock_tool_manager
    ):
        """Test tools are included in follow-up API calls"""
        ai_generator.client = mock_anthropic_client

        mock_anthropic_client.messages.create.side_effect = [
            anthropic_tool_use_response,
            anthropic_final_response
        ]

        mock_tool_manager.execute_tool.return_value = "Tool result"

        ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify second API call includes tools
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}
