"""LangGraph state machine for the Acme Dental booking agent."""

from langgraph.graph import END, StateGraph

# Nodes
from src.nodes import (
    ask_for_name_email,
    ask_for_time_preference,
    booking_check_availability,
    booking_collect_identity,
    booking_create,
    check_existing_flow,
    confirm_action,
    confirm_booking,
    handle_faq,
    lookup_events,
    parse_slot_selection,
    parse_time_preference,
    respond_to_user,
    router,
    select_event,
    tool_error_handler,
)

# Routing
from src.routing import (
    route_after_availability_check,
    route_after_identity_check,
    route_after_lookup,
    route_after_router,
    route_after_selection,
    route_after_slot_selection,
    route_from_entry,
)
from src.state import AgentState


def create_booking_graph():
    """Create and compile the booking agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_existing_flow", check_existing_flow)
    workflow.add_node("router", router)
    workflow.add_node("booking_collect_identity", booking_collect_identity)
    workflow.add_node("ask_for_name_email", ask_for_name_email)
    workflow.add_node("ask_for_time_preference", ask_for_time_preference)
    workflow.add_node("parse_time_preference", parse_time_preference)
    workflow.add_node("booking_check_availability", booking_check_availability)
    workflow.add_node("parse_slot_selection", parse_slot_selection)

    workflow.add_node("booking_create", booking_create)
    workflow.add_node("confirm_booking", confirm_booking)

    # Generic nodes (Book/Cancel/Reschedule)
    workflow.add_node("lookup_events", lookup_events)
    workflow.add_node("select_event", select_event)
    workflow.add_node("confirm_action", confirm_action)

    workflow.add_node("handle_faq", handle_faq)
    workflow.add_node("respond_to_user", respond_to_user)
    workflow.add_node("tool_error_handler", tool_error_handler)

    # Set entry point to check state first
    workflow.set_entry_point("check_existing_flow")

    # Add edges
    workflow.add_conditional_edges("check_existing_flow", route_from_entry)

    workflow.add_conditional_edges("router", route_after_router)

    workflow.add_conditional_edges("booking_collect_identity", route_after_identity_check)

    workflow.add_edge("ask_for_name_email", END)

    workflow.add_edge("ask_for_time_preference", END)
    workflow.add_edge("parse_time_preference", "booking_check_availability")

    workflow.add_conditional_edges("booking_check_availability", route_after_availability_check)

    workflow.add_conditional_edges("parse_slot_selection", route_after_slot_selection)

    workflow.add_edge("booking_create", "confirm_booking")
    workflow.add_edge("confirm_booking", END)

    # Generic flow edges
    workflow.add_conditional_edges("lookup_events", route_after_lookup)
    workflow.add_conditional_edges("select_event", route_after_selection)
    workflow.add_edge("confirm_action", END)

    workflow.add_edge("handle_faq", END)
    workflow.add_edge("respond_to_user", END)
    workflow.add_edge("tool_error_handler", END)

    return workflow.compile()
