from typing import Literal

from src.state import AgentState


def route_from_entry(
    state: AgentState,
) -> Literal[
    "router",
    "booking_collect_identity",
    "lookup_events",
    "select_event",
    "confirm_action",
    "ask_for_time_preference",
]:
    """Entry point routing based on flow and intent."""
    flow = state.get("flow", "IDLE")
    intent = state.get("intent")

    # If we are in an active flow, route accordingly
    if flow == "BOOK":
        return "booking_collect_identity"

    if flow == "CANCEL":
        # If we have selected an event, we are waiting for confirmation
        if state.get("selected_event_uri"):
            return "confirm_action"
        # If we have looked up events but not selected, we are choosing
        if state.get("matched_events"):
            return "select_event"
        # Otherwise start lookup
        return "lookup_events"

    if flow == "RESCHEDULE":
        # If we have selected event, proceed to new booking details
        if state.get("selected_event_uri"):
            # If we already have a slot selected, we might be in booking creation,
            # but usually entry point routing implies we are waiting for user input.
            # We route to booking controller to handle identity check and next steps.
            return "booking_collect_identity"

        if state.get("matched_events"):
            return "select_event"

        return "lookup_events"

    # IDLE flow - check intent from previous turn or default to router
    if intent == "BOOK":
        return "booking_collect_identity"
    if intent == "CANCEL":
        return "lookup_events"
    if intent == "RESCHEDULE":
        return "lookup_events"

    # Default to router classification
    return "router"


def route_after_router(
    state: AgentState,
) -> Literal["handle_faq", "booking_collect_identity", "lookup_events", "respond_to_user"]:
    """Route based on classified intent."""
    intent = state.get("intent", "GENERAL")

    if intent == "BOOK":
        return "booking_collect_identity"
    elif intent == "CANCEL":
        return "lookup_events"
    elif intent == "RESCHEDULE":
        return "lookup_events"
    elif intent == "FAQ":
        return "handle_faq"
    else:
        return "respond_to_user"


def route_after_identity_check(
    state: AgentState,
) -> Literal[
    "ask_for_name_email",
    "ask_for_time_preference",
    "parse_time_preference",
    "booking_check_availability",
    "parse_slot_selection",
    "booking_create",
]:
    """Check if we have name/email and route appropriately."""
    has_identity = state.get("user_name") and state.get("user_email")
    # has_preference = state.get("time_preference") is not None
    # Wait, existing code:
    asked_for_preference = state.get("asked_for_preference", False)
    has_preference = state.get("time_preference") is not None
    has_slots = state.get("available_slots") is not None
    has_selection = state.get("selected_slot") is not None

    # CRITICAL: If user selected a slot, create the booking!
    if has_selection:
        return "booking_create"

    # If we have identity
    if has_identity:
        # If we've shown slots but user hasn't selected, parse their selection
        if has_slots and not has_selection:
            return "parse_slot_selection"
        # If we asked for preference but haven't parsed yet, parse it now
        if asked_for_preference and not has_preference:
            return "parse_time_preference"
        # If we have preference but no slots, check availability
        if has_preference and not has_slots:
            return "booking_check_availability"
        # If no preference collected yet, ask for it
        if not has_preference and not asked_for_preference:
            return "ask_for_time_preference"

    # Missing identity info
    return "ask_for_name_email"


def route_after_availability_check(state: AgentState) -> Literal["tool_error_handler", "__end__"]:
    """Check if availability check succeeded - slots already presented."""
    if state.get("error"):
        return "tool_error_handler"
    return "__end__"


def route_after_slot_selection(state: AgentState) -> Literal["booking_create", "parse_time_preference", "__end__"]:
    """Route after slot selection attempt."""
    if state.get("selected_slot"):
        return "booking_create"

    # If slots were cleared (because user typed text), go re-parse preference
    if state.get("available_slots") is None:
        return "parse_time_preference"

    # If no valid selection but slots exist (and we sent error msg), END
    return "__end__"


def route_after_lookup(state: AgentState) -> Literal["__end__"]:
    """Route after looking up bookings."""
    # Always wait for user input (confirmation or selection)
    return "__end__"


def route_after_selection(state: AgentState) -> Literal["ask_for_time_preference", "confirm_action", "__end__"]:
    """Route based on flow after event selection."""
    flow = state.get("flow")
    if not state.get("selected_event_uri"):
        return "__end__"

    if flow == "RESCHEDULE":
        return "ask_for_time_preference"
    elif flow == "CANCEL":
        # Wait for user confirmation (handled in next turn via route_from_entry -> confirm_action)
        return "__end__"

    return "__end__"
