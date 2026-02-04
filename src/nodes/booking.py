import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.calendly_client import CalendlyClient
from src.llm import get_llm
from src.state import AgentState
from src.tools import create_booking


def booking_collect_identity(state: AgentState) -> AgentState:
    """
    Extract user's name and email from conversation history.

    Strategy:
    1. Check state for existing identity.
    2. Try regex extraction for "name email" patterns.
    3. Use LLM extraction if regex fails or is ambiguous.
    """

    # If we already have both, skip extraction
    if state.get("user_name") and state.get("user_email"):
        return state

    # Get the last user message for regex extraction
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    # Try simple regex first (faster and more reliable for "name email" format)

    text = last_user_msg.content
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, text)

    regex_name = None
    regex_email = None

    if emails:
        regex_email = emails[0]
        # Name is everything before the email
        name_part = text.split(regex_email)[0].strip()
        if name_part and len(name_part) > 2:
            regex_name = name_part

    llm = get_llm(temperature=0)

    system_prompt = """Extract ONLY the user's full name and email address from the conversation.

Respond in this exact JSON format:
{"name": "John Doe", "email": "john@example.com"}

If either is missing:
{"name": null, "email": null}

IMPORTANT:
- Only extract if explicitly stated
- Do NOT ask follow-up questions
- Do NOT make assumptions
- Return ONLY the JSON, nothing else"""

    response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])

    try:
        # Clean response in case LLM added extra text
        content = response.content.strip()
        # Find JSON object in response
        if "{" in content:
            json_start = content.index("{")
            json_end = content.rindex("}") + 1
            content = content[json_start:json_end]

        data = json.loads(content)
        name = data.get("name") or regex_name  # Prefer LLM, fallback to regex
        email = data.get("email") or regex_email

        # Only update if we found new values (don't overwrite with None)
        updates = {}
        if name:
            updates["user_name"] = name
        if email:
            updates["user_email"] = email

        return {**state, **updates}
    except Exception:
        # Use regex results if LLM failed
        updates = {}
        if regex_name:
            updates["user_name"] = regex_name
        if regex_email:
            updates["user_email"] = regex_email
        return {**state, **updates} if updates else state


def ask_for_time_preference(state: AgentState) -> AgentState:
    """
    Prompt user for a preferred day or time.

    Sets 'asked_for_preference' flag to True to track conversation state.
    """
    msg = (
        "Do you have a preferred day or time for your appointment? "
        "For example, 'Tuesday afternoon' or 'Wednesday morning'. Or just say 'any' if you're flexible!"
    )
    return {**state, "messages": [AIMessage(content=msg)], "asked_for_preference": True}


def parse_time_preference(state: AgentState) -> AgentState:
    """
    Parse natural language time preferences into structured data.

    Extracts:
    - Days of week (Monday, Tuesday...)
    - Specific hours (11am, 2pm)
    - General times (morning, afternoon, evening)

    Falls back to 'any' if input is ambiguous like "anytime".
    """
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    user_input = last_user_msg.content.strip().lower()

    # Check if user says "any"
    if any(word in user_input for word in ["any", "anytime", "flexible"]):
        return {**state, "time_preference": "any"}

    # Extract ALL days mentioned
    days_map = {
        "monday": ["monday", "mon"],
        "tuesday": ["tuesday", "tues", "tue"],
        "wednesday": ["wednesday", "wed"],
        "thursday": ["thursday", "thurs", "thu"],
        "friday": ["friday", "fri"],
        "saturday": ["saturday", "sat"],
        "sunday": ["sunday", "sun"],
    }

    found_days = []
    for day, keywords in days_map.items():
        if any(kw in user_input for kw in keywords):
            found_days.append(day)

    # Extract specific time (11 am, 2:30 pm, etc.)
    time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    time_matches = re.findall(time_pattern, user_input)

    specific_hour = None
    if time_matches:
        hour_str, minute_str, period = time_matches[0]
        hour = int(hour_str)
        # minute = int(minute_str) if minute_str else 0  # Unused

        # Convert to 24-hour
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        specific_hour = hour

    # Extract general time
    general_time = None
    if not specific_hour:
        if "morning" in user_input or "am" in user_input:
            general_time = "morning"
        elif "afternoon" in user_input or "pm" in user_input:
            general_time = "afternoon"
        elif "evening" in user_input:
            general_time = "evening"

    # Build preference: "tuesday,wednesday|11" or "monday|morning" or "any"
    if not found_days and not specific_hour and not general_time:
        return {**state, "time_preference": "any"}

    pref_parts = []
    if found_days:
        pref_parts.append(",".join(found_days))
    if specific_hour is not None:
        pref_parts.append(f"hour:{specific_hour}")
    elif general_time:
        pref_parts.append(general_time)

    preference = "|".join(pref_parts) if pref_parts else "any"
    print(f"📅 User preference: {preference}")
    return {**state, "time_preference": preference}


def ask_for_name_email(state: AgentState) -> AgentState:
    """Ask user to provide missing name or email."""
    missing = []
    if not state.get("user_name"):
        missing.append("full name")
    if not state.get("user_email"):
        missing.append("email address")

    msg = f"To complete your booking, I'll need your {' and '.join(missing)}. Could you please provide that?"

    return {**state, "messages": [AIMessage(content=msg)]}


def booking_check_availability(state: AgentState) -> AgentState:
    """
    Fetch slots from Calendly and filter based on user preferences.

    Logic:
    1. Fetch all available slots for the event type.
    2. Parse 'time_preference' string (e.g. "monday,tuesday|morning").
    3. Filter slots to find Exact Matches and Fallback Matches.
    4. Limit results to preventing overwhelming the user.
    """
    try:
        client = CalendlyClient()
        event_types = client.get_event_types()
        event_type_uri = event_types[0]["uri"]

        slots = client.get_available_times(event_type_uri)

        if not slots:
            return {**state, "error": "No available slots found."}

        # Parse preference: "tuesday,wednesday|hour:11" or "monday|morning" or "any"
        preference = state.get("time_preference", "any")

        if preference == "any":
            display_slots = slots[:10]
            msg_header = "Here are the next available appointment slots:"
        else:
            # Parse preference components
            parts = preference.split("|")
            requested_days = parts[0].split(",") if parts[0] else []

            specific_hour = None
            general_time = None
            if len(parts) > 1:
                if parts[1].startswith("hour:"):
                    specific_hour = int(parts[1].split(":")[1])
                else:
                    general_time = parts[1]

            # Try exact match first (specific day + specific hour)
            exact_matches = []
            fallback_matches = []

            for slot in slots:
                try:
                    dt = datetime.fromisoformat(slot["start_time"].replace("Z", "+00:00"))
                    day_name = dt.strftime("%A").lower()
                    hour = dt.hour

                    # Check day match
                    day_match = not requested_days or day_name in requested_days

                    if day_match:
                        if specific_hour is not None:
                            # Exact hour match
                            if hour == specific_hour:
                                exact_matches.append(slot)
                            # Fallback: morning/afternoon/evening based on requested hour
                            elif specific_hour < 12 and 6 <= hour < 12:
                                fallback_matches.append(slot)
                            elif 12 <= specific_hour < 17 and 12 <= hour < 17:
                                fallback_matches.append(slot)
                        elif general_time:
                            # General time match
                            if (
                                (general_time == "morning" and 6 <= hour < 12)
                                or (general_time == "afternoon" and 12 <= hour < 17)
                                or (general_time == "evening" and 17 <= hour < 21)
                            ):
                                exact_matches.append(slot)
                        else:
                            # Just day match
                            exact_matches.append(slot)
                except Exception:
                    continue

            # Decide what to show
            if exact_matches:
                display_slots = exact_matches[:10]
                days_str = ", ".join(requested_days) if requested_days else "any day"
                if specific_hour is not None:
                    time_str = f"{specific_hour % 12 or 12} {'AM' if specific_hour < 12 else 'PM'}"
                    msg_header = f"Here are slots for {days_str} at {time_str}:"
                elif general_time:
                    msg_header = f"Here are {general_time} slots for {days_str}:"
                else:
                    msg_header = f"Here are slots for {days_str}:"
            elif fallback_matches:
                display_slots = fallback_matches[:10]
                days_str = ", ".join(requested_days) if requested_days else "those days"
                time_str = f"{specific_hour % 12 or 12} {'AM' if specific_hour < 12 else 'PM'}"
                fallback_time = "morning" if specific_hour < 12 else "afternoon"
                msg_header = f"I don't have {time_str} available, but here are {fallback_time} times for {days_str}:"
            else:
                # No matches at all - show all available
                display_slots = slots[:10]
                msg_header = "No slots found for your preference. Here are all available times:"

        # Format slots
        formatted_slots = []
        for i, slot in enumerate(display_slots, 1):
            start_time = slot.get("start_time", "")
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%A, %B %d at %I:%M %p UTC")
                formatted_slots.append(f"{i}. {formatted_time}")
            except Exception:
                formatted_slots.append(f"{i}. {start_time}")

        msg = f"{msg_header}\n\n{chr(10).join(formatted_slots)}\n\nReply with the slot number to book (e.g., '2')."

        return {**state, "messages": [AIMessage(content=msg)], "available_slots": display_slots, "error": None}
    except Exception as e:
        return {**state, "error": f"ERROR: {str(e)}"}


def parse_slot_selection(state: AgentState) -> AgentState:
    """
    Parse user input to identify selected slot number.

    Handles:
    - Valid integer input (1-N): Sets 'selected_slot'.
    - Invalid integer (out of range): Returns error message to retry.
    - Text input (e.g. "actually tuesday"): Clears 'available_slots' to trigger re-parsing of preference.
    """
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg or not state.get("available_slots"):
        return state

    user_input = last_user_msg.content.strip()

    # Try to parse as a number
    try:
        slot_num = int(user_input)
        if 1 <= slot_num <= len(state["available_slots"]):
            selected = state["available_slots"][slot_num - 1]
            selected_time = selected.get("start_time")
            print(f"✅ User selected slot {slot_num}: {selected_time}")
            # DON'T add a message - just set the slot and let routing handle it
            return {**state, "selected_slot": selected_time}
        else:
            msg = f"Please choose a number between 1 and {len(state['available_slots'])}."
            return {**state, "messages": [AIMessage(content=msg)]}
    except ValueError:
        # User didn't type a number - assume they are changing preference
        # We clear available_slots to trigger re-route to parsing
        return {**state, "available_slots": None, "messages": []}


def booking_create(state: AgentState) -> AgentState:
    """
    Execute the booking creation via Calendly API.

    Features:
    - Validates required fields (slot, name, email).
    - Creates the new appointment.
    - If in RESCHEDULE flow, attempts to cancel the old appointment ('selected_event_uri').
    - Returns success message and resets flow to IDLE.
    """
    if not state.get("selected_slot") or not state.get("user_name") or not state.get("user_email"):
        return {**state, "booking_stage": "ERROR", "error": "Missing required information for booking"}

    try:
        result = create_booking.invoke(
            {"start_time": state["selected_slot"], "name": state["user_name"], "email": state["user_email"]}
        )

        if result.startswith("ERROR"):
            return {**state, "booking_stage": "ERROR", "error": result}

        # Check if this was a reschedule -> Cancel the old event
        reschedule_msg = ""
        # Check unified flow and URI
        if state.get("flow") == "RESCHEDULE" and state.get("selected_event_uri"):
            try:
                # Cancel the old event
                client = CalendlyClient()
                _ = client.cancel_event(state["selected_event_uri"], reason="Rescheduled to new time")
                reschedule_msg = "\n\n♻️ Your previous appointment has been successfully cancelled."
            except Exception as e:
                reschedule_msg = (
                    f"\n\n⚠️ Warning: I booked your new appointment, but failed to cancel the old one. Error: {str(e)}"
                )

        # SUCCESS: Clear booking state to prevent re-routing loops
        # Reset flow to IDLE
        return {
            **state,
            "messages": [AIMessage(content=result + reschedule_msg)],
            "flow": "IDLE",
            "intent": None,
            "selected_slot": None,
            "available_slots": None,
            "time_preference": None,
            "asked_for_preference": False,
            "error": None,
            "matched_events": None,
            "selected_event_uri": None,
        }
    except Exception as e:
        return {**state, "booking_stage": "ERROR", "error": f"ERROR: {str(e)}"}


def confirm_booking(state: AgentState) -> AgentState:
    """Generate booking confirmation message."""
    # Confirmation is handled in booking_create
    return state
