import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

from src.calendly_client import CalendlyClient
from src.state import AgentState


def lookup_events(state: AgentState) -> AgentState:
    """
    Search for existing appointments for Cancel/Reschedule flows.

    Logic:
    1. Identify lookup email (New input > Stored lookup > User profile).
    2. If missing, prompt user to provide email.
    3. Call Calendly API to list scheduled events.
    4. Handle valid bookings (1 or multiple) or failure (suggest retry).

    CRITICAL: Does NOT auto-select single event anymore (waits for explicit confirmation).
    """
    flow = state.get("flow")

    # Check for new email in user message first (allows correction)
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    new_email = None
    if last_user_msg:
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, last_user_msg.content)
        if emails:
            new_email = emails[0]

    # Prioritize new email, then stored lookup, then user profile
    email = new_email or state.get("lookup_email") or state.get("user_email")

    if not email:
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if not last_user_msg:
            return state

        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, last_user_msg.content)

        if not emails:
            action = "reschedule" if flow == "RESCHEDULE" else "cancel"
            msg = f"To {action} your appointment, please provide your email address."
            return {**state, "messages": [AIMessage(content=msg)]}

        email = emails[0]

    try:
        client = CalendlyClient()
        bookings = client.list_scheduled_events(email)

        if not bookings:
            msg = f"I couldn't find any upcoming appointments for {email}."
            # Don't persist invalid email if we just found it
            return {**state, "messages": [AIMessage(content=msg)], "lookup_email": None if new_email else email}

        # Format bookings
        booking_list = []
        for i, booking in enumerate(bookings, 1):
            start_time = booking.get("start_time", "")
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%A, %B %d at %I:%M %p")
                booking_list.append(f"{i}. {formatted_time}")
            except Exception:
                booking_list.append(f"{i}. {start_time}")

        action_prompt = "cancel" if flow == "CANCEL" else "reschedule"

        if len(bookings) == 1:
            appt_time = booking_list[0].split(". ", 1)[1]
            msg = f"Found your appointment on **{appt_time}**.\n\nIs this the one you want to {action_prompt}? (Yes/No)"
        else:
            msg = "Found these appointments:\n\n" + "\n".join(booking_list)
            msg += f"\n\nWhich one would you like to {action_prompt}? Reply with the number (1-{len(bookings)})."

        return {
            **state,
            "messages": [AIMessage(content=msg)],
            "lookup_email": None if new_email else email,
            "matched_events": bookings,
            "user_email": email,
            # CRITICAL: Do NOT set selected_event_uri yet, wait for user confirmation in select_event
            "selected_event_uri": None,
        }
    except Exception as e:
        msg = f"Error looking up appointments: {str(e)}"
        return {**state, "messages": [AIMessage(content=msg)], "error": str(e)}


def select_event(state: AgentState) -> AgentState:
    """
    Process user selection from a list of looked-up appointments.

    Handles:
    - Yes/No confirmation for single booking.
    - Number selection (1-N) for multiple bookings.
    - Sets 'selected_event_uri' upon valid selection.
    - Transitions flow:
      - CANCEL -> confirm_action
      - RESCHEDULE -> booking flow (clears old slots/preferences)
    """
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    response = last_user_msg.content.strip().lower()
    bookings = state.get("matched_events", [])
    flow = state.get("flow")

    selected_booking = None

    if len(bookings) == 1:
        if response in ["yes", "y", "sure", "correct"]:
            selected_booking = bookings[0]
        else:
            msg = "Okay, let me know if you need anything else."
            return {**state, "messages": [AIMessage(content=msg)], "intent": None, "flow": "IDLE"}
    else:
        # Multiple bookings
        try:
            selection = int(response)
            if 1 <= selection <= len(bookings):
                selected_booking = bookings[selection - 1]
            else:
                msg = f"Please choose a number between 1 and {len(bookings)}."
                return {**state, "messages": [AIMessage(content=msg)]}
        except ValueError:
            msg = f"Please reply with a number (1-{len(bookings)}) to select the appointment."
            return {**state, "messages": [AIMessage(content=msg)]}

    if selected_booking:
        uri = selected_booking["uri"]

        if flow == "CANCEL":
            # Ask for final confirmation
            start_time = selected_booking.get("start_time", "")
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%A, %B %d at %I:%M %p")
            except Exception:
                formatted_time = start_time

            msg = f"You selected: {formatted_time}\n\nAre you sure you want to cancel this appointment? (Yes/No)"
            return {**state, "messages": [AIMessage(content=msg)], "selected_event_uri": uri}

        elif flow == "RESCHEDULE":
            # Transition to booking flow
            user_name = selected_booking.get("name") or state.get("user_name")
            msg = (
                "Got it. I will cancel the old one only AFTER we confirm the new one.\n\n"
                "When would you like to reschedule this to? (e.g. 'Tuesday morning' or 'Feb 5th')"
            )
            return {
                **state,
                "messages": [AIMessage(content=msg)],
                "selected_event_uri": uri,
                "user_name": user_name,
                "time_preference": None,
                "available_slots": None,
                "selected_slot": None,
            }

    return state


def confirm_action(state: AgentState) -> AgentState:
    """Confirm and execute cancellation."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    response = last_user_msg.content.strip().lower()

    if response in ["yes", "y", "confirm", "sure"]:
        # Execute Cancellation
        if not state.get("selected_event_uri"):
            return state

        try:
            client = CalendlyClient()
            _ = client.cancel_event(state["selected_event_uri"])
            msg = "✅ Your appointment has been successfully canceled."
            return {
                **state,
                "messages": [AIMessage(content=msg)],
                "flow": "IDLE",
                "intent": None,
                "matched_events": None,
                "selected_event_uri": None,
                "confirmed": True,
            }
        except Exception as e:
            msg = f"Error cancelling appointment: {str(e)}"
            return {**state, "messages": [AIMessage(content=msg)], "error": str(e)}
    else:
        msg = "No problem! Your appointment remains scheduled."
        return {
            **state,
            "messages": [AIMessage(content=msg)],
            "flow": "IDLE",
            "intent": None,
            "matched_events": None,
            "selected_event_uri": None,
        }
