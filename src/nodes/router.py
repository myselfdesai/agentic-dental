from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.state import AgentState


def check_existing_flow(state: AgentState) -> AgentState:
    """Entry node - check if we're already in a booking flow."""
    flow = state.get("flow", "IDLE")

    # If IDLE, standard routing applies
    if flow == "IDLE":
        return state

    # If in active flow, check for interrupts (intent changes)
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_user_msg:
        content_lower = last_user_msg.content.lower()

        # Keywords (must match router logic)
        cancel_keywords = ["cancel", "cancellation", "delete", "remove"]
        book_keywords = ["book", "appointment", "schedule", "slot", "available", "checkup", "check-up"]

        new_flow = None
        new_intent = None

        # Check CANCEL
        if any(keyword in content_lower for keyword in cancel_keywords):
            new_flow = "CANCEL"
            new_intent = "CANCEL"

        # Check RESCHEDULE (params override Book)
        elif "reschedule" in content_lower or "change" in content_lower or "move" in content_lower:
            new_flow = "RESCHEDULE"
            new_intent = "RESCHEDULE"

        # Check BOOK
        elif any(keyword in content_lower for keyword in book_keywords):
            new_flow = "BOOK"
            new_intent = "BOOK"

        # If we detected a valid flow change, apply it
        if new_flow and new_flow != flow:
            # We specifically want to allow switching, so we return the new state
            # which will be picked up by route_from_entry
            return {**state, "flow": new_flow, "intent": new_intent}

    # If no interrupt, continue existing flow
    return state


def router(state: AgentState) -> AgentState:
    """Classify user intent: BOOK, FAQ, or GENERAL."""

    # Quick keyword check first - if user says "book" or "appointment", it's BOOK
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_user_msg:
        content_lower = last_user_msg.content.lower()
        book_keywords = ["book", "appointment", "schedule", "slot", "available", "checkup", "check-up"]
        cancel_keywords = ["cancel", "cancellation", "delete", "remove"]

        if any(keyword in content_lower for keyword in cancel_keywords):
            return {**state, "intent": "CANCEL", "flow": "CANCEL"}

        # Check RESCHEDULE before BOOK to prevent "schedule" keyword conflict
        # "reschedule" contains "schedule", so specific check first
        if "reschedule" in content_lower or "change" in content_lower or "move" in content_lower:
            return {**state, "intent": "RESCHEDULE", "flow": "RESCHEDULE"}

        if any(keyword in content_lower for keyword in book_keywords) or content_lower in ["yes", "y", "sure", "yeah"]:
            return {**state, "intent": "BOOK", "flow": "BOOK"}

    llm = get_llm(temperature=0)

    system_prompt = """Classify user intent:
- BOOK: mentions booking, appointment, slots, availability
- CANCEL: mentions cancel, cancellation, delete appointment
- RESCHEDULE: mentions reschedule, change appointment time, move appointment
- FAQ: asks about prices, hours, services, policies
- GENERAL: greetings only

ONE WORD: BOOK, CANCEL, RESCHEDULE, FAQ, or GENERAL"""

    if not last_user_msg:
        return {**state, "intent": "GENERAL"}

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_msg.content)])

    intent = response.content.strip().upper()
    if intent not in ["BOOK", "CANCEL", "RESCHEDULE", "FAQ", "GENERAL"]:
        intent = "GENERAL"

    # Sync flow with intent
    flow = "IDLE"
    if intent == "BOOK":
        flow = "BOOK"
    elif intent == "CANCEL":
        flow = "CANCEL"
    elif intent == "RESCHEDULE":
        flow = "RESCHEDULE"

    return {**state, "intent": intent, "flow": flow}
