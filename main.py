#!/usr/bin/env python3
"""Entry point for the Basic Agent application.

This module provides a clean entry point that initializes the agent
with proper error handling and configuration management.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import AgentError, BasicAgent, ConfigurationError, ModelError, load_config
from src.models import select_and_initialize_model


def main() -> None:
    """Main entry point with comprehensive error handling."""
    try:
        # Load configuration
        config = load_config()

        # Initialize model
        model_name, llm = select_and_initialize_model()
        config.lm_studio_model = model_name

        # Create and run agent
        agent = BasicAgent(llm, config, model_name)
        agent.run()

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except ModelError as e:
        print(f"Model error: {e}")
        sys.exit(1)
    except AgentError as e:
        print(f"Agent error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\nShutdown requested by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
