"""State definition for the Booking Agent."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the booking agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str | None
    flow: Literal["IDLE", "BOOK", "CANCEL", "RESCHEDULE"]

    # User Identity
    user_name: str | None
    user_email: str | None

    # Booking data
    time_preference: str | None
    asked_for_preference: bool
    available_slots: list[dict] | None
    selected_slot: str | None

    # Event data (Cancel/Reschedule)
    lookup_email: str | None
    matched_events: list[dict] | None
    selected_event_uri: str | None
    confirmed: bool

    error: str | None
