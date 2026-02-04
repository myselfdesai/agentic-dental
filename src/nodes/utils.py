from langchain_core.messages import AIMessage

from src.state import AgentState


def tool_error_handler(state: AgentState) -> AgentState:
    """Handle errors from tool calls."""
    error = state.get("error", "Unknown error occurred")

    msg = (
        f"I apologize, but I encountered an issue: {error}\n\n"
        "Would you like to try again or can I help you with something else?"
    )

    return {**state, "messages": [AIMessage(content=msg)], "error": None}
