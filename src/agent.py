"""Main agent implementation with modular architecture."""

from typing import Any

import lmstudio as lms

from .config import AgentConfig
from .exceptions import AgentError, ToolError, UserCancellationError
from .memory import MemoryManager
from .tools import AVAILABLE_TOOLS


class BasicAgent:
    """Main agent class with clean architecture and comprehensive error handling."""

    def __init__(self, llm: lms.LLM, config: AgentConfig):
        """Initialize the agent with configuration and dependencies.

        Args:
            llm: Initialized LM Studio LLM instance
            config: Agent configuration
        """
        self.llm = llm
        self.config = config
        self.memory = MemoryManager(config.memory_long_term_path, config.memory_short_term_cap)
        self.tools = dict(AVAILABLE_TOOLS)

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
            full_prompt = f"{user_input}\n{memory_context}" if memory_context else user_input

            # Convert tool objects to functions for LLM
            available_tool_functions = self._prepare_tool_functions()

            print("Thinking...")

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
                max_prediction_rounds=3,  # Limit rounds to prevent infinite loops
            )

            # Combine all response parts
            full_response = "".join(response_parts) if response_parts else "No response generated"

            print(f"\nAgent Response: {full_response}")

            # Update memory
            self.memory.update_memory(user_input, full_response)

        except UserCancellationError:
            print("Operation cancelled by user.")
        except ToolError as e:
            print(f"Tool error: {e}")
        except Exception as e:
            raise AgentError(f"Error processing input: {e}") from e

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
