from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for searching content and retrieving course structure.

Available Tools:
1. **search_course_content** - Search within course lesson content for specific information, concepts, or answers
2. **get_course_outline** - Retrieve course structure, lesson lists, and navigation information

Tool Selection Guidelines:
- **Use search_course_content for**:
  - Questions about specific concepts, topics, or technical details
  - "What is X?", "How does Y work?", "Explain Z"
  - Content-focused queries requiring information from lesson materials

- **Use get_course_outline for**:
  - Questions about course structure, organization, or what lessons are available
  - "Show me the outline", "What lessons are in X?", "What's the structure of Y?"
  - Navigation and discovery queries about course organization

- **Use no tools for**:
  - General knowledge questions unrelated to the specific course materials
  - Conversational queries or clarifications

Tool Usage Constraints:
- **Maximum two sequential tool calls per query** (you may call tools multiple times if needed, but use at most 2 rounds)
- Choose the most appropriate tool based on query intent
- If tool yields no results, state this clearly without offering unrelated alternatives

Response Protocol:
- **No meta-commentary**: Provide direct answers without mentioning:
  - "Based on the search results..."
  - "I searched the course materials..."
  - "Using the outline tool..."
  - Your reasoning or tool selection process
- **Be concise**: Get to the point quickly while maintaining educational value
- **Be clear**: Use accessible language appropriate for learners
- **Include examples**: When they aid understanding, provide relevant examples
- **Synthesize effectively**: Combine tool results into coherent, natural responses

Remember: Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls with support for sequential rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        from config import config

        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response

        # Loop for up to MAX_TOOL_ROUNDS
        for round_num in range(config.MAX_TOOL_ROUNDS):
            # Exit if no tool use
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

            # Add AI's tool use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # CRITICAL: Include tools in next API call for sequential rounds
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get(
                    "tools"
                ),  # Include tools (was excluded before)
                "tool_choice": {"type": "auto"},
            }

            # Get next response
            current_response = self.client.messages.create(**next_params)

        # Handle max rounds exceeded
        if current_response.stop_reason == "tool_use":
            # Try to extract any text content from tool_use response
            text_blocks = [
                block.text
                for block in current_response.content
                if getattr(block, "type", None) == "text"
            ]
            return text_blocks[0] if text_blocks else "Maximum tool rounds reached."

        return current_response.content[0].text
