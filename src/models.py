"""Model management for LM Studio integration."""

import json
import os
import subprocess
from typing import Any

import lmstudio as lms

from .exceptions import ModelError


def list_available_models() -> list[dict[str, str]]:
    """List available models using LM Studio CLI.

    Returns:
        List of model dictionaries with metadata

    Raises:
        ModelError: If unable to list models
    """
    try:
        result = subprocess.run(
            ["lms", "ls", "--json"], capture_output=True, text=True, check=True, timeout=30
        )
        models = json.loads(result.stdout)
        return models
    except subprocess.CalledProcessError as e:
        raise ModelError(
            "Failed to list models. Ensure LM Studio is installed and the 'lms' command is available."
        ) from e
    except json.JSONDecodeError as e:
        raise ModelError(f"Failed to parse model list: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ModelError("Timeout while listing models") from e


def get_model_info(model_key: str) -> dict[str, Any]:
    """Get detailed information about a specific model.
    
    Args:
        model_key: The model key to look up
        
    Returns:
        Dictionary with model information including tool capability
        
    Raises:
        ModelError: If model not found or lookup fails
    """
    try:
        models = list_available_models()
        for model in models:
            if model.get("modelKey") == model_key:
                return model
        raise ModelError(f"Model '{model_key}' not found")
    except Exception as e:
        raise ModelError(f"Failed to get model info: {e}") from e


def filter_llm_models(models: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter models to only include LLM models (not embedding models).

    Args:
        models: List of all available models

    Returns:
        List of LLM models only
    """
    return [model for model in models if model.get("type") == "llm"]


def select_model_interactive(models: list[dict[str, str]]) -> str:
    """Interactive model selection with graceful error handling.

    Args:
        models: List of available LLM models

    Returns:
        Selected model key

    Raises:
        ModelError: If no models available or user cancels
    """
    if not models:
        raise ModelError(
            "No LLM models available. Please download an LLM model in LM Studio first."
        )

    # Check for environment variable override
    auto_model = os.getenv("LMS_MODEL")
    if auto_model:
        for model in models:
            if model["modelKey"] == auto_model:
                print(f"Using model from environment: {model['displayName']}")
                return auto_model
        print(f"Warning: Model '{auto_model}' from environment not found, showing selection menu")

    # Interactive selection
    print("Available LLM Models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model['displayName']} ({model['modelKey']})")

    default_model = models[0]["modelKey"]
    print(f"\\nPress Enter for default ({models[0]['displayName']}) or select a number:")

    while True:
        try:
            user_input = input("Select a model by number (or Enter for default): ").strip()

            if not user_input:
                print(f"Using default model: {models[0]['displayName']}")
                return default_model

            selected_index = int(user_input) - 1
            if 0 <= selected_index < len(models):
                return models[selected_index]["modelKey"]
            else:
                print(f"Please enter a number between 1 and {len(models)}")

        except ValueError:
            print("Please enter a valid number or press Enter for default")
        except (KeyboardInterrupt, EOFError):
            print(f"\\nUsing default model: {models[0]['displayName']}")
            return default_model


def ensure_server_running() -> None:
    """Start LM Studio server if not already running.

    Raises:
        ModelError: If unable to manage server
    """
    try:
        server_status = subprocess.run(
            ["lms", "server", "status"], capture_output=True, text=True, check=True, timeout=10
        )

        if "not running" in server_status.stdout:
            print("Starting LM Studio server...")
            subprocess.run(["lms", "server", "start"], check=True, timeout=30)
            print("Server started successfully")

    except subprocess.CalledProcessError as e:
        raise ModelError(f"Failed to manage LM Studio server: {e}") from e
    except subprocess.TimeoutExpired:
        raise ModelError("Timeout while managing LM Studio server")


def initialize_llm(model_name: str) -> lms.LLM:
    """Initialize LM Studio model instance.

    Args:
        model_name: Model key to initialize

    Returns:
        Initialized LLM instance

    Raises:
        ModelError: If model initialization fails
    """
    try:
        return lms.llm(model_name)
    except Exception as e:
        raise ModelError(f"Failed to initialize model '{model_name}': {e}") from e


def select_and_initialize_model() -> tuple[str, lms.LLM]:
    """Complete model selection and initialization flow.

    Returns:
        Tuple of (model_name, llm_instance)

    Raises:
        ModelError: If any step fails
    """
    print("=== Model Setup ===")

    # List and filter models
    all_models = list_available_models()
    llm_models = filter_llm_models(all_models)

    # Select model
    selected_model = select_model_interactive(llm_models)
    print(f"Selected model: {selected_model}")

    # Ensure server is running
    ensure_server_running()

    # Initialize model
    print("Initializing model...")
    llm = initialize_llm(selected_model)

    return selected_model, llm
