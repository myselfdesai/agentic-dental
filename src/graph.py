"""LangGraph state machine for the Acme Dental booking agent."""

import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.calendly_client import CalendlyClient
from src.tools import create_booking, retrieve_faq
from src.llm import get_llm



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

        if any(keyword in content_lower for keyword in book_keywords):
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

    # Parse response
    import json

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
        from datetime import datetime

        from src.calendly_client import CalendlyClient

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
                _ = client.cancel_event(
                    state["selected_event_uri"],
                    reason="Rescheduled to new time"
                )
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
        import re
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

        import re
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
            appt_time = booking_list[0].split('. ', 1)[1]
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
             return {
                 **state,
                 "messages": [AIMessage(content=msg)],
                 "selected_event_uri": uri
             }

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
                "confirmed": True
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
             "selected_event_uri": None
        }

# Remove old nodes











def handle_faq(state: AgentState) -> AgentState:
    """Handle FAQ queries using RAG."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    try:
        # Retrieve FAQ context
        context = retrieve_faq.invoke({"query": last_user_msg.content})

        # Generate response with context
        llm = get_llm(temperature=0.3)
        system_prompt = f"""You are a helpful assistant for Acme Dental clinic.

Use the following knowledge base information to answer the user's question:

{context}

Provide a clear, concise, and answer based on the knowledge base.
If the information isn't in the knowledge base, politely say you don't have that information."""

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_msg.content)])

        return {**state, "messages": [AIMessage(content=response.content)]}
    except Exception as e:
        return {**state, "error": f"ERROR: {str(e)}"}


def respond_to_user(state: AgentState) -> AgentState:
    """Final response node for general conversation."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state

    llm = get_llm(temperature=0.3)
    system_prompt = """You are a concise dental receptionist for Acme Dental.

Respond in 1-2 sentences MAX. Be friendly but brief.
If they're just greeting, greet back and ask: "Would you like to book an appointment or have questions?"
Do NOT make up information. Do NOT offer services we don't have."""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_msg.content)])

    return {**state, "messages": [AIMessage(content=response.content)]}


def tool_error_handler(state: AgentState) -> AgentState:
    """Handle errors from tool calls."""
    error = state.get("error", "Unknown error occurred")

    msg = (
        f"I apologize, but I encountered an issue: {error}\n\n"
        "Would you like to try again or can I help you with something else?"
    )

    return {**state, "messages": [AIMessage(content=msg)], "error": None}


def route_from_entry(
    state: AgentState,
) -> Literal[
    "router",
    "booking_collect_identity",
    "lookup_events",
    "select_event",
    "confirm_action",
    "ask_for_time_preference"
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
) -> Literal[
    "handle_faq",
    "booking_collect_identity",
    "lookup_events",
    "respond_to_user"
]:
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
    has_preference = state.get("time_preference") is not None
    asked_for_preference = state.get("asked_for_preference", False)
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

def route_after_selection(state: AgentState) -> Literal[
    "ask_for_time_preference",
    "confirm_action",
    "__end__"
]:
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
