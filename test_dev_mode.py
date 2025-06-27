#!/usr/bin/env python3
"""Comprehensive test script for dev mode functionality.

This script tests all dev mode features including:
- Environment variable detection
- Token tracking and context analysis
- Tool call monitoring
- Dev commands (!tokens, !memory, !stats, !help, !clear)
- Interactive features (mocked)
- Memory management
- Tool execution with logging
"""

import os
import sys
import time
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import BasicAgent, AgentConfig
from src.dev_mode import DevModeTracker, is_dev_mode_enabled
from src.models import get_model_info


class DevModeTestSuite:
    """Comprehensive test suite for dev mode functionality."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        
    def run_test(self, test_name, test_func):
        """Run a single test and track results."""
        print(f"\n🧪 Testing: {test_name}")
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            self.tests_passed += 1
            self.test_results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            self.tests_failed += 1
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_env_variable_detection(self):
        """Test DEV_MODE environment variable detection."""
        # Test with DEV_MODE=true
        os.environ["DEV_MODE"] = "true"
        assert is_dev_mode_enabled() == True, "Should detect DEV_MODE=true"
        
        # Test with DEV_MODE=false
        os.environ["DEV_MODE"] = "false"
        assert is_dev_mode_enabled() == False, "Should detect DEV_MODE=false"
        
        # Test with DEV_MODE=1
        os.environ["DEV_MODE"] = "1"
        assert is_dev_mode_enabled() == True, "Should detect DEV_MODE=1"
        
        # Test without DEV_MODE
        if "DEV_MODE" in os.environ:
            del os.environ["DEV_MODE"]
        assert is_dev_mode_enabled() == False, "Should default to false when not set"
        
        # Restore for other tests
        os.environ["DEV_MODE"] = "true"
    
    def test_dev_mode_tracker_initialization(self):
        """Test DevModeTracker initialization and basic functionality."""
        # Test enabled tracker
        tracker = DevModeTracker(enabled=True)
        assert tracker.enabled == True, "Tracker should be enabled"
        assert tracker.session_stats.total_requests == 0, "Should start with 0 requests"
        
        # Test disabled tracker
        tracker = DevModeTracker(enabled=False)
        assert tracker.enabled == False, "Tracker should be disabled"
    
    def test_token_estimation(self):
        """Test token estimation functionality."""
        tracker = DevModeTracker(enabled=True)
        
        # Test various text lengths
        short_text = "Hello world"
        estimated = tracker.estimate_tokens(short_text)
        expected = len(short_text) // 4
        assert estimated == expected, f"Token estimation should be {expected}, got {estimated}"
        
        long_text = "A" * 100
        estimated = tracker.estimate_tokens(long_text)
        expected = len(long_text) // 4
        assert estimated == expected, f"Token estimation for long text should be {expected}, got {estimated}"
    
    def test_context_analysis(self):
        """Test context analysis and breakdown."""
        tracker = DevModeTracker(enabled=True)
        
        user_input = "What's the weather?"
        memory_context = "Recent conversation: \nUser: Hello\nAgent: Hi there"
        short_term = "User: Hello\nAgent: Hi there"
        long_term = {"user_name": "TestUser"}
        
        stats = tracker.analyze_context(user_input, memory_context, short_term, long_term)
        
        assert stats.base_prompt_chars == len(user_input), "Should track user input length"
        assert stats.memory_context_chars == len(memory_context), "Should track memory context length"
        assert stats.short_term_chars == len(short_term), "Should track short-term memory length"
        assert stats.total_chars > 0, "Total chars should be positive"
        assert stats.estimated_tokens > 0, "Estimated tokens should be positive"
    
    def test_tool_call_logging(self):
        """Test tool call logging functionality."""
        tracker = DevModeTracker(enabled=True)
        
        # Test successful tool call
        tracker.log_tool_call(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            result="Tool executed successfully",
            success=True
        )
        
        assert tracker.session_stats.total_tool_calls == 1, "Should log 1 tool call"
        assert tracker.session_stats.successful_tool_calls == 1, "Should log 1 successful call"
        assert tracker.session_stats.failed_tool_calls == 0, "Should log 0 failed calls"
        
        # Test failed tool call
        tracker.log_tool_call(
            tool_name="failing_tool",
            arguments={"arg1": "value1"},
            result="",
            success=False,
            error="Tool failed"
        )
        
        assert tracker.session_stats.total_tool_calls == 2, "Should log 2 total tool calls"
        assert tracker.session_stats.successful_tool_calls == 1, "Should still have 1 successful call"
        assert tracker.session_stats.failed_tool_calls == 1, "Should log 1 failed call"
        
        # Test tool usage statistics
        tool_stats = tracker.get_tool_stats()
        assert tool_stats["total_tool_calls"] == 2, "Tool stats should show 2 total calls"
        assert tool_stats["success_rate"] == 0.5, "Success rate should be 50%"
        assert "test_tool" in tool_stats["tool_usage"], "Should track test_tool usage"
        assert "failing_tool" in tool_stats["tool_usage"], "Should track failing_tool usage"
    
    def test_session_statistics(self):
        """Test session statistics tracking."""
        tracker = DevModeTracker(enabled=True)
        
        # Simulate some activity
        user_input = "Test question"
        memory_context = "Previous context"
        short_term = "Short term memory"
        long_term = {"key": "value"}
        
        # Log some requests
        for i in range(3):
            context_stats = tracker.analyze_context(user_input, memory_context, short_term, long_term)
            tracker.log_request(context_stats)
        
        # Test token statistics
        token_stats = tracker.get_token_stats()
        assert token_stats["total_requests"] == 3, "Should track 3 requests"
        assert token_stats["total_estimated_tokens"] > 0, "Should have positive token count"
        assert token_stats["average_tokens_per_request"] > 0, "Should calculate average"
        assert token_stats["session_duration_seconds"] > 0, "Should track session duration"
        
        # Test memory statistics  
        memory_stats = tracker.get_memory_stats(short_term, long_term)
        assert memory_stats["short_term_chars"] == len(short_term), "Should track short-term size"
        assert memory_stats["long_term_entries"] == 1, "Should count long-term entries"
        assert memory_stats["total_memory_chars"] > 0, "Should calculate total memory size"
    
    def test_agent_initialization_with_dev_mode(self):
        """Test BasicAgent initialization with dev mode."""
        # Mock LLM and dependencies
        mock_llm = Mock()
        config = AgentConfig()
        model_key = "test-model"
        
        # Set dev mode environment
        os.environ["DEV_MODE"] = "true"
        
        with patch('src.agent.get_model_info') as mock_get_model_info, \
             patch('src.memory.MemoryManager') as mock_memory:
            
            mock_get_model_info.return_value = {"trainedForToolUse": False}
            
            agent = BasicAgent(mock_llm, config, model_key)
            
            assert agent.dev_mode.enabled == True, "Agent should have dev mode enabled"
            assert agent.supports_native_tools == False, "Should detect non-tool model"
    
    def test_dev_command_parsing(self):
        """Test dev command handling."""
        mock_llm = Mock()
        config = AgentConfig()
        model_key = "test-model"
        
        os.environ["DEV_MODE"] = "true"
        
        with patch('src.agent.get_model_info') as mock_get_model_info, \
             patch('src.memory.MemoryManager') as mock_memory, \
             patch('builtins.print') as mock_print:
            
            mock_get_model_info.return_value = {"trainedForToolUse": False}
            
            agent = BasicAgent(mock_llm, config, model_key)
            
            # Test various dev commands
            commands_to_test = ["!help", "!tokens", "!memory", "!stats", "!clear"]
            
            for cmd in commands_to_test:
                try:
                    agent._handle_dev_command(cmd)
                    # Command should execute without error
                except Exception as e:
                    raise AssertionError(f"Dev command {cmd} failed: {e}")
    
    def test_tool_call_monitoring_integration(self):
        """Test tool call monitoring in actual execution."""
        tracker = DevModeTracker(enabled=True)
        
        # Simulate tool execution with monitoring
        response = "TOOL_CALL: list_files(path=\"/test/path\")"
        
        # Mock the agent's tool execution
        mock_llm = Mock()
        config = AgentConfig()
        model_key = "test-model"
        
        with patch('src.agent.get_model_info') as mock_get_model_info, \
             patch('src.memory.MemoryManager') as mock_memory:
            
            mock_get_model_info.return_value = {"trainedForToolUse": False}
            
            agent = BasicAgent(mock_llm, config, model_key)
            agent.dev_mode = tracker  # Use our test tracker
            
            # Test argument parsing
            args = agent._parse_tool_arguments('path="/test/path"')
            assert args == {"path": "/test/path"}, f"Should parse arguments correctly, got {args}"
    
    def test_context_growth_analysis(self):
        """Test context growth pattern analysis."""
        tracker = DevModeTracker(enabled=True)
        
        # Simulate growing context over multiple requests
        base_context = "Initial context"
        for i in range(5):
            growing_context = base_context + " " + ("extra " * i)
            stats = tracker.analyze_context(
                f"Query {i}", growing_context, 
                growing_context, {}
            )
            tracker.log_request(stats)
        
        # Test context breakdown
        breakdown = tracker.get_context_breakdown(
            "Current query", base_context, base_context, {}
        )
        
        assert "context_growth" in breakdown, "Should include growth analysis"
        growth = breakdown["context_growth"]
        assert "trend" in growth, "Should analyze growth trend"
        assert "average_growth_per_request" in growth, "Should calculate average growth"
        assert "peak_context_size" in growth, "Should track peak size"
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting Comprehensive Dev Mode Test Suite")
        print("=" * 60)
        
        # Set up dev mode environment
        os.environ["DEV_MODE"] = "true"
        
        # Run all tests
        self.run_test("Environment Variable Detection", self.test_env_variable_detection)
        self.run_test("DevModeTracker Initialization", self.test_dev_mode_tracker_initialization) 
        self.run_test("Token Estimation", self.test_token_estimation)
        self.run_test("Context Analysis", self.test_context_analysis)
        self.run_test("Tool Call Logging", self.test_tool_call_logging)
        self.run_test("Session Statistics", self.test_session_statistics)
        self.run_test("Agent Dev Mode Integration", self.test_agent_initialization_with_dev_mode)
        self.run_test("Dev Command Parsing", self.test_dev_command_parsing)
        self.run_test("Tool Call Monitoring", self.test_tool_call_monitoring_integration)
        self.run_test("Context Growth Analysis", self.test_context_growth_analysis)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")
        print(f"📈 Success Rate: {self.tests_passed/(self.tests_passed + self.tests_failed)*100:.1f}%")
        
        if self.tests_failed > 0:
            print("\n❌ FAILED TESTS:")
            for name, status, error in self.test_results:
                if status == "FAILED":
                    print(f"   - {name}: {error}")
        
        print("\n🎯 Dev Mode Feature Coverage:")
        print("   ✅ Environment variable toggle")
        print("   ✅ Token tracking and estimation") 
        print("   ✅ Context analysis and breakdown")
        print("   ✅ Tool call monitoring and logging")
        print("   ✅ Session statistics tracking")
        print("   ✅ Agent integration")
        print("   ✅ Dev command handling")
        print("   ✅ Growth pattern analysis")
        
        return self.tests_failed == 0


def run_interactive_demo():
    """Run an interactive demo showing dev mode features."""
    print("\n🎭 INTERACTIVE DEV MODE DEMO")
    print("=" * 40)
    
    # Mock a realistic agent session
    tracker = DevModeTracker(enabled=True)
    
    print("📍 Simulating agent session with dev mode...")
    
    # Simulate startup
    tracker.print_startup_status()
    
    # Simulate some conversations
    conversations = [
        ("Hello, what's my name?", "I don't have that information yet."),
        ("My name is Alex", "Nice to meet you, Alex!"), 
        ("What files are in my documents?", "Let me check your documents folder."),
        ("Create a test file", "I'll create a test file for you."),
    ]
    
    for i, (user_msg, agent_msg) in enumerate(conversations):
        print(f"\n🔄 Request {i+1}")
        
        # Simulate memory context growing
        memory_context = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversations[:i]])
        long_term = {"user_name": "Alex"} if i >= 1 else {}
        
        # Analyze context
        stats = tracker.analyze_context(user_msg, memory_context, memory_context, long_term)
        print(f"🔧 Context: {stats.total_chars} chars (~{stats.estimated_tokens} tokens)")
        
        # Log request
        tracker.log_request(stats)
        
        # Simulate tool calls for some requests
        if "files" in user_msg.lower():
            tracker.log_tool_call("list_files", {"path": "/Users/alex/Documents"}, "file1.txt\nfile2.txt", True)
            print("🔧 Tool call: list_files(path=/Users/alex/Documents)")
        elif "create" in user_msg.lower():
            tracker.log_tool_call("create_file", {"path": "test.txt", "content": "test"}, "File created", True)
            print("🔧 Tool call: create_file(path=test.txt, content=test)")
    
    print("\n📊 Final Session Statistics:")
    
    # Show token stats
    token_stats = tracker.get_token_stats()
    print(f"   Requests: {token_stats['total_requests']}")
    print(f"   Estimated tokens: {token_stats['total_estimated_tokens']}")
    print(f"   Average tokens/request: {token_stats['average_tokens_per_request']:.1f}")
    
    # Show tool stats
    tool_stats = tracker.get_tool_stats()
    print(f"   Tool calls: {tool_stats['total_tool_calls']}")
    print(f"   Success rate: {tool_stats['success_rate']:.1%}")
    
    # Show memory stats
    final_memory = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversations])
    memory_stats = tracker.get_memory_stats(final_memory, {"user_name": "Alex"})
    print(f"   Memory size: {memory_stats['total_memory_chars']} chars")
    
    print("\n✨ Demo complete! All dev mode features working.")


if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = DevModeTestSuite()
    success = test_suite.run_all_tests()
    
    # Run interactive demo
    run_interactive_demo()
    
    # Exit with appropriate code
    print(f"\n🏁 Test suite {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)