"""Test basic agent functionality."""

import os
import tempfile
from unittest.mock import Mock, patch

# Import from the new modular structure
from src import AgentConfig, BasicAgent
from src.memory import MemoryManager


def test_memory_functions():
    """Test memory loading and saving functions."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        # Test memory manager
        memory_manager = MemoryManager(temp_path)

        # Test initial state
        assert memory_manager.long_term_memory == {}

        # Test saving and loading
        memory_manager.long_term_memory = {"test_key": "test_value"}
        memory_manager.save_long_term_memory()

        # Create new instance to test loading
        new_memory_manager = MemoryManager(temp_path)
        assert new_memory_manager.long_term_memory == {"test_key": "test_value"}
    finally:
        os.unlink(temp_path)


def test_agent_initialization():
    """Test agent initialization."""
    mock_llm = Mock()
    config = AgentConfig()

    with patch("src.memory.MemoryManager"), patch("src.agent.get_model_info"):
        agent = BasicAgent(mock_llm, config, "test-model")
        assert agent.llm == mock_llm
        assert agent.config == config


def test_memory_update():
    """Test memory update functionality."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        memory_manager = MemoryManager(temp_path, short_term_cap=50)

        # Test normal update
        memory_manager.update_memory("Hello", "Hi there")
        expected = "\nUser: Hello\nAgent: Hi there"
        assert memory_manager.short_term_memory == expected

        # Test truncation
        long_input = "A" * 100
        long_response = "B" * 100
        memory_manager.update_memory(long_input, long_response)

        # Should be truncated to last 50 characters
        assert len(memory_manager.short_term_memory) == 50
    finally:
        os.unlink(temp_path)
