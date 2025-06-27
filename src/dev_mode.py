"""Development mode observability and debugging tools."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextStats:
    """Statistics about context usage."""
    total_chars: int = 0
    estimated_tokens: int = 0
    short_term_chars: int = 0
    long_term_chars: int = 0
    base_prompt_chars: int = 0
    memory_context_chars: int = 0


@dataclass
class ToolCallLog:
    """Log entry for a tool call."""
    timestamp: float
    tool_name: str
    arguments: Dict[str, Any]
    result_preview: str
    success: bool
    error: Optional[str] = None


@dataclass
class SessionStats:
    """Comprehensive session statistics."""
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    total_estimated_tokens: int = 0
    total_prompt_tokens: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    context_history: List[ContextStats] = field(default_factory=list)
    tool_call_history: List[ToolCallLog] = field(default_factory=list)


class DevModeTracker:
    """Comprehensive development mode observability tracker."""
    
    def __init__(self, enabled: bool = False):
        """Initialize dev mode tracker.
        
        Args:
            enabled: Whether dev mode is active
        """
        self.enabled = enabled
        self.session_stats = SessionStats()
        self._chars_per_token = 4  # Rough estimate
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Ensure at least 1 token for non-empty text
        return max(1, len(text) // self._chars_per_token)
    
    def analyze_context(self, user_input: str, memory_context: str, 
                       short_term_memory: str, long_term_memory: Dict[str, Any]) -> ContextStats:
        """Analyze and break down context components.
        
        Args:
            user_input: Current user input
            memory_context: Full memory context string
            short_term_memory: Short-term memory content
            long_term_memory: Long-term memory dictionary
            
        Returns:
            Context statistics breakdown
        """
        long_term_str = json.dumps(long_term_memory) if long_term_memory else ""
        
        stats = ContextStats(
            short_term_chars=len(short_term_memory),
            long_term_chars=len(long_term_str),
            memory_context_chars=len(memory_context),
            base_prompt_chars=len(user_input)
        )
        
        stats.total_chars = stats.base_prompt_chars + stats.memory_context_chars
        # Calculate tokens from the actual total text length
        total_text = user_input + memory_context
        stats.estimated_tokens = self.estimate_tokens(total_text)
        
        return stats
    
    def log_request(self, context_stats: ContextStats) -> None:
        """Log a request with context statistics.
        
        Args:
            context_stats: Context breakdown for this request
        """
        if not self.enabled:
            return
            
        self.session_stats.total_requests += 1
        self.session_stats.total_estimated_tokens += context_stats.estimated_tokens
        self.session_stats.total_prompt_tokens += context_stats.estimated_tokens
        self.session_stats.context_history.append(context_stats)
        
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], 
                     result: str, success: bool, error: Optional[str] = None) -> None:
        """Log a tool call execution.
        
        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the call succeeded
            error: Error message if failed
        """
        if not self.enabled:
            return
            
        # Create result preview (first 200 chars)
        result_preview = result[:200] + "..." if len(result) > 200 else result
        
        tool_log = ToolCallLog(
            timestamp=time.time(),
            tool_name=tool_name,
            arguments=arguments,
            result_preview=result_preview,
            success=success,
            error=error
        )
        
        self.session_stats.tool_call_history.append(tool_log)
        self.session_stats.total_tool_calls += 1
        
        if success:
            self.session_stats.successful_tool_calls += 1
        else:
            self.session_stats.failed_tool_calls += 1
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.session_stats.start_time
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics.
        
        Returns:
            Dictionary with token statistics
        """
        return {
            "total_requests": self.session_stats.total_requests,
            "total_estimated_tokens": self.session_stats.total_estimated_tokens,
            "average_tokens_per_request": (
                self.session_stats.total_estimated_tokens / max(1, self.session_stats.total_requests)
            ),
            "session_duration_seconds": self.get_session_duration(),
            "tokens_per_minute": (
                self.session_stats.total_estimated_tokens / max(1, self.get_session_duration() / 60)
            )
        }
    
    def get_memory_stats(self, short_term_memory: str, 
                        long_term_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get current memory statistics.
        
        Args:
            short_term_memory: Current short-term memory
            long_term_memory: Current long-term memory
            
        Returns:
            Dictionary with memory statistics
        """
        long_term_str = json.dumps(long_term_memory) if long_term_memory else ""
        
        return {
            "short_term_chars": len(short_term_memory),
            "short_term_tokens": self.estimate_tokens(short_term_memory),
            "long_term_chars": len(long_term_str),
            "long_term_tokens": self.estimate_tokens(long_term_str),
            "long_term_entries": len(long_term_memory),
            "total_memory_chars": len(short_term_memory) + len(long_term_str),
            "total_memory_tokens": self.estimate_tokens(short_term_memory + long_term_str)
        }
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool call statistics.
        
        Returns:
            Dictionary with tool call statistics
        """
        tool_usage = {}
        for log_entry in self.session_stats.tool_call_history:
            tool_name = log_entry.tool_name
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {"calls": 0, "successes": 0, "failures": 0}
            
            tool_usage[tool_name]["calls"] += 1
            if log_entry.success:
                tool_usage[tool_name]["successes"] += 1
            else:
                tool_usage[tool_name]["failures"] += 1
        
        return {
            "total_tool_calls": self.session_stats.total_tool_calls,
            "successful_calls": self.session_stats.successful_tool_calls,
            "failed_calls": self.session_stats.failed_tool_calls,
            "success_rate": (
                self.session_stats.successful_tool_calls / max(1, self.session_stats.total_tool_calls)
            ),
            "tool_usage": tool_usage
        }
    
    def get_context_breakdown(self, user_input: str, memory_context: str,
                            short_term_memory: str, long_term_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed context breakdown.
        
        Args:
            user_input: Current user input
            memory_context: Full memory context
            short_term_memory: Short-term memory content
            long_term_memory: Long-term memory dictionary
            
        Returns:
            Detailed context analysis
        """
        stats = self.analyze_context(user_input, memory_context, short_term_memory, long_term_memory)
        
        return {
            "current_request": {
                "user_input_chars": len(user_input),
                "user_input_tokens": self.estimate_tokens(user_input),
                "memory_context_chars": stats.memory_context_chars,
                "memory_context_tokens": self.estimate_tokens(memory_context),
                "total_context_chars": stats.total_chars,
                "total_context_tokens": stats.estimated_tokens
            },
            "memory_breakdown": {
                "short_term_chars": stats.short_term_chars,
                "short_term_tokens": self.estimate_tokens(short_term_memory),
                "long_term_chars": stats.long_term_chars,
                "long_term_tokens": self.estimate_tokens(json.dumps(long_term_memory) if long_term_memory else "")
            },
            "context_growth": self._analyze_context_growth()
        }
    
    def _analyze_context_growth(self) -> Dict[str, Any]:
        """Analyze context growth patterns over the conversation."""
        if len(self.session_stats.context_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_contexts = self.session_stats.context_history[-5:]  # Last 5 requests
        
        # Calculate growth trend
        total_chars = [ctx.total_chars for ctx in recent_contexts]
        if len(total_chars) >= 2:
            avg_growth = sum(total_chars[i] - total_chars[i-1] for i in range(1, len(total_chars))) / (len(total_chars) - 1)
        else:
            avg_growth = 0
        
        return {
            "trend": "growing" if avg_growth > 50 else "stable" if avg_growth > -50 else "shrinking",
            "average_growth_per_request": avg_growth,
            "current_context_size": total_chars[-1] if total_chars else 0,
            "peak_context_size": max(ctx.total_chars for ctx in self.session_stats.context_history) if self.session_stats.context_history else 0
        }
    
    def print_startup_status(self) -> None:
        """Print dev mode status on startup."""
        if self.enabled:
            print("🔧 DEV MODE ENABLED")
            print("   - Token tracking active")
            print("   - Tool call monitoring active") 
            print("   - Interactive prompt inspection available")
            print("   - Dev commands enabled (!tokens, !memory, !stats, !help)")
            print("   - Context analysis active")
            print()


def is_dev_mode_enabled() -> bool:
    """Check if dev mode is enabled via environment variable.
    
    Returns:
        True if DEV_MODE environment variable is set to 'true'
    """
    return os.getenv("DEV_MODE", "").lower() in ("true", "1", "yes", "on")