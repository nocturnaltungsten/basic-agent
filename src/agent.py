"""Main agent implementation with modular architecture."""

import os
from typing import Any

import lmstudio as lms

from .config import AgentConfig
from .exceptions import AgentError, ToolError, UserCancellationError
from .memory import MemoryManager
from .models import get_model_info
from .tools import AVAILABLE_TOOLS


class BasicAgent:
    """Main agent class with clean architecture and comprehensive error handling."""

    def __init__(self, llm: lms.LLM, config: AgentConfig, model_key: str):
        """Initialize the agent with configuration and dependencies.

        Args:
            llm: Initialized LM Studio LLM instance
            config: Agent configuration
            model_key: The model key for capability detection
        """
        self.llm = llm
        self.config = config
        self.model_key = model_key
        self.memory = MemoryManager(config.memory_long_term_path, config.memory_short_term_cap)
        self.tools = dict(AVAILABLE_TOOLS)
        
        # Detect tool capability
        self.supports_native_tools = self._detect_tool_capability()

    def process_user_input(self, user_input: str) -> None:
        """Process a single user input through the agent loop.

        Args:
            user_input: User's input message

        Raises:
            AgentError: If processing fails
        """
        try:
            # Prepare context with memory
            memory_context = self.memory.get_memory_context()
            
            print("Thinking...")

            if self.supports_native_tools:
                # Use native tool calling for tool-trained models
                full_response = self._process_with_native_tools(user_input, memory_context)
            else:
                # Use prompt-based approach for other models
                full_response = self._process_with_prompt_tools(user_input, memory_context)

            print(f"\nAgent Response: {full_response}")

            # Update memory
            self.memory.update_memory(user_input, full_response)

        except UserCancellationError:
            print("Operation cancelled by user.")
        except ToolError as e:
            print(f"Tool error: {e}")
        except Exception as e:
            raise AgentError(f"Error processing input: {e}") from e

    def _process_with_native_tools(self, user_input: str, memory_context: str) -> str:
        """Process input using native tool calling for tool-trained models."""
        full_prompt = f"{user_input}\n{memory_context}" if memory_context else user_input
        available_tool_functions = self._prepare_tool_functions()
        
        # Collect response parts, filtering out tool call metadata
        response_parts = []

        def on_message(message: object) -> None:
            """Handle messages from LLM, filtering out tool call metadata."""
            if hasattr(message, "content"):
                if isinstance(message.content, str):
                    if not self._is_tool_metadata(message.content):
                        response_parts.append(message.content)
                elif hasattr(message.content, "__iter__"):
                    for item in message.content:
                        if hasattr(item, "text"):
                            text = item.text
                            if not self._is_tool_metadata(text):
                                response_parts.append(text)
                        else:
                            text = str(item)
                            if not self._is_tool_metadata(text):
                                response_parts.append(text)
                else:
                    text = str(message.content)
                    if not self._is_tool_metadata(text):
                        response_parts.append(text)
            elif isinstance(message, str):
                if not self._is_tool_metadata(message):
                    response_parts.append(message)

        # Execute with tools
        self.llm.act(
            full_prompt,
            tools=available_tool_functions,
            on_message=on_message,
            max_prediction_rounds=3,
        )

        return "".join(response_parts) if response_parts else "No response generated"

    def _process_with_prompt_tools(self, user_input: str, memory_context: str) -> str:
        """Process input using prompt-based tool calling for non-tool-trained models."""
        import re
        import json
        
        # Create tool descriptions for the prompt
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(f"- {name}: {tool.description}")
        
        tools_text = "\n".join(tool_descriptions)
        
        # Enhanced prompt for tool usage
        enhanced_prompt = f"""You are a helpful AI assistant with access to the following tools:

{tools_text}

To use a tool, respond with EXACTLY this format:
TOOL_CALL: tool_name(argument1="value1", argument2="value2")

For example:
- To list files: TOOL_CALL: list_files(path="/Users/username/Documents")
- To run terminal commands: TOOL_CALL: terminal(command="ls -la")

Current working directory: {os.getcwd()}

User request: {user_input}
{f"Context: {memory_context}" if memory_context else ""}

Response:"""

        # Get LLM response
        prediction_result = self.llm.respond(enhanced_prompt)
        
        # Extract text from PredictionResult
        if hasattr(prediction_result, 'content'):
            response = str(prediction_result.content)
        elif hasattr(prediction_result, 'text'):
            response = str(prediction_result.text)
        else:
            response = str(prediction_result)
        
        # Parse and execute tool calls
        return self._parse_and_execute_tool_calls(response)

    def _parse_and_execute_tool_calls(self, response: str) -> str:
        """Parse tool calls from LLM response and execute them."""
        import re
        import json
        
        # Look for TOOL_CALL: patterns
        tool_call_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(tool_call_pattern, response)
        
        if not matches:
            # No tool calls found, return original response
            return response
            
        result_parts = []
        remaining_response = response
        
        for tool_name, args_str in matches:
            if tool_name in self.tools:
                try:
                    # Parse arguments
                    args = self._parse_tool_arguments(args_str)
                    
                    # Execute tool
                    tool_result = self.tools[tool_name].execute(**args)
                    result_parts.append(f"Tool {tool_name} result: {tool_result}")
                    
                    # Remove the tool call from response
                    tool_call_text = f"TOOL_CALL: {tool_name}({args_str})"
                    remaining_response = remaining_response.replace(tool_call_text, "").strip()
                    
                except Exception as e:
                    result_parts.append(f"Error executing {tool_name}: {e}")
            else:
                result_parts.append(f"Unknown tool: {tool_name}")
        
        # Combine remaining response with tool results
        if remaining_response and remaining_response != response:
            return f"{remaining_response}\n\n{chr(10).join(result_parts)}"
        else:
            return chr(10).join(result_parts)

    def _parse_tool_arguments(self, args_str: str) -> dict:
        """Parse tool arguments from string format."""
        import re
        
        args = {}
        # Parse key="value" patterns
        arg_pattern = r'(\w+)=(["\'])(.*?)\2'
        matches = re.findall(arg_pattern, args_str)
        
        for key, quote, value in matches:
            args[key] = value
            
        return args

    def _prepare_tool_functions(self) -> list:
        """Convert tool objects to functions that LM Studio can call."""
        tool_functions = []

        for name, tool in self.tools.items():
            # Create a function that wraps the tool's execute method
            def create_tool_function(tool_obj, tool_name):
                def tool_function(**kwargs):
                    return tool_obj.execute(**kwargs)

                # Set function name and docstring for LM Studio
                tool_function.__name__ = tool_name
                tool_function.__doc__ = tool_obj.description
                return tool_function

            tool_functions.append(create_tool_function(tool, name))

        return tool_functions

    def _detect_tool_capability(self) -> bool:
        """Detect if the current model supports native tool calling."""
        try:
            model_info = get_model_info(self.model_key)
            supports_tools = model_info.get("trainedForToolUse", False)
            print(f"Model {self.model_key}: Native tool support = {supports_tools}")
            return supports_tools
        except Exception as e:
            print(f"Could not detect tool capability: {e}")
            print("Defaulting to prompt-based approach")
            return False

    def _is_tool_metadata(self, text: str) -> bool:
        """Check if text is tool call metadata that should be filtered out."""
        return text.startswith("ToolCallRequestData") or text.startswith("ToolCallResultData")

    def run(self) -> None:
        """Run the main agent interaction loop."""
        print("Basic Agent started. Type 'quit' or 'exit' to stop.")
        print("Type your message and press Enter.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                self.process_user_input(user_input)
                print()  # Add spacing

            except KeyboardInterrupt:
                print("\\nGoodbye!")
                break
            except EOFError:
                print("\\nGoodbye!")
                break
            except AgentError as e:
                print(f"Agent error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get agent status and statistics.

        Returns:
            Dictionary with agent status information
        """
        return {
            "model": self.config.lm_studio_model,
            "tools_available": list(self.tools.keys()),
            "memory_stats": self.memory.get_stats(),
        }
