"""LangGraph state machine for the Acme Dental booking agent."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.tools import check_availability, create_booking, retrieve_faq


class AgentState(TypedDict):
    """State for the booking agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str | None
    booking_stage: Literal["IDLE", "NEED_IDENTITY", "NEED_PREFERENCE", "SHOWING_SLOTS", "BOOKING", "CONFIRMED", "ERROR"]
    user_name: str | None
    user_email: str | None
    time_preference: str | None
    asked_for_preference: bool  # Keep for now, will phase out
    selected_slot: str | None
    available_slots: list[dict] | None
    error: str | None


def check_existing_flow(state: AgentState) -> AgentState:
    """Entry node - check if we're already in a booking flow."""
    # If we already classified as BOOK and have name/email, continue booking
    # Don't re-route on every message!
    return state


def router(state: AgentState) -> AgentState:
    """Classify user intent: BOOK, FAQ, or GENERAL."""
    
    # Quick keyword check first - if user says "book" or "appointment", it's BOOK
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_user_msg:
        content_lower = last_user_msg.content.lower()
        book_keywords = ["book", "appointment", "schedule", "slot", "available", "checkup", "check-up"]
        if any(keyword in content_lower for keyword in book_keywords):
            print(f"🔍 Router: Quick match = BOOK")
            return {**state, "intent": "BOOK"}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = """Classify user intent:
- BOOK: mentions booking, appointment, slots, availability
- FAQ: asks about prices, hours, services, policies  
- GENERAL: greetings only

ONE WORD: BOOK, FAQ, or GENERAL"""
    
    if not last_user_msg:
        return {**state, "intent": "GENERAL"}
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_msg.content)
    ])
    
    intent = response.content.strip().upper()
    if intent not in ["BOOK", "FAQ", "GENERAL"]:
        intent = "GENERAL"
    
    return {**state, "intent": intent}


def booking_collect_identity(state: AgentState) -> AgentState:
    """Extract name and email from conversation. Don't overwrite if already collected."""
    
    # If we already have both, skip extraction
    if state.get("user_name") and state.get("user_email"):
        return state
    
    # Get the last user message for regex extraction
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state
    
    # Try simple regex first (faster and more reliable for "name email" format)
    import re
    text = last_user_msg.content
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    regex_name = None
    regex_email = None
    
    if emails:
        regex_email = emails[0]
        # Name is everything before the email
        name_part = text.split(regex_email)[0].strip()
        if name_part and len(name_part) > 2:
            regex_name = name_part
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
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
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
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
    except Exception as e:
        # Use regex results if LLM failed
        updates = {}
        if regex_name:
            updates["user_name"] = regex_name
        if regex_email:
            updates["user_email"] = regex_email
        return {**state, **updates} if updates else state


def ask_for_time_preference(state: AgentState) -> AgentState:
    """Ask user for their preferred day or time."""
    msg = "Do you have a preferred day or time for your appointment? For example, 'Tuesday afternoon' or 'Wednesday morning'. Or just say 'any' if you're flexible!"
    return {**state, "messages": [AIMessage(content=msg)], "asked_for_preference": True}


def parse_time_preference(state: AgentState) -> AgentState:
    """Parse user's time preference: extract days and specific/general time."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state
    
    user_input = last_user_msg.content.strip().lower()
    
    # Check if user says "any"
    if any(word in user_input for word in ["any", "anytime", "flexible"]):
        return {**state, "time_preference": "any"}
    
    import re
    
    # Extract ALL days mentioned
    days_map = {
        "monday": ["monday", "mon"],
        "tuesday": ["tuesday", "tues", "tue"],
        "wednesday": ["wednesday", "wed"],
        "thursday": ["thursday", "thurs", "thu"],
        "friday": ["friday", "fri"],
        "saturday": ["saturday", "sat"],
        "sunday": ["sunday", "sun"]
    }
    
    found_days = []
    for day, keywords in days_map.items():
        if any(kw in user_input for kw in keywords):
            found_days.append(day)
    
    # Extract specific time (11 am, 2:30 pm, etc.)
    time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
    time_matches = re.findall(time_pattern, user_input)
    
    specific_hour = None
    if time_matches:
        hour_str, minute_str, period = time_matches[0]
        hour = int(hour_str)
        minute = int(minute_str) if minute_str else 0
        
        # Convert to 24-hour
        if period == 'pm' and hour != 12:
            hour += 12
        elif period == 'am' and hour == 12:
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
    """Check slots and filter by preference with clear fallback messaging."""
    try:
        from src.calendly_client import CalendlyClient
        from datetime import datetime
        
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
                            if ((general_time == "morning" and 6 <= hour < 12) or
                                (general_time == "afternoon" and 12 <= hour < 17) or
                                (general_time == "evening" and 17 <= hour < 21)):
                                exact_matches.append(slot)
                        else:
                            # Just day match
                            exact_matches.append(slot)
                except:
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
            except:
                formatted_slots.append(f"{i}. {start_time}")
        
        msg = f"{msg_header}\n\n{chr(10).join(formatted_slots)}\n\nReply with the slot number to book (e.g., '2')."
        
        return {**state, "messages": [AIMessage(content=msg)], "available_slots": display_slots, "error": None}
    except Exception as e:
        return {**state, "error": f"ERROR: {str(e)}"}


def parse_slot_selection(state: AgentState) -> AgentState:
    """Parse user's slot selection from their message."""
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
        # User didn't type a number
        msg = f"Please reply with a slot number (1-{len(state['available_slots'])}) to book your appointment."
        return {**state, "messages": [AIMessage(content=msg)]}





def booking_create(state: AgentState) -> AgentState:
    """Create the booking via Calendly."""
    if not state.get("selected_slot") or not state.get("user_name") or not state.get("user_email"):
        return {**state, "booking_stage": "ERROR", "error": "Missing required information for booking"}
    
    try:
        result = create_booking.invoke({
            "start_time": state["selected_slot"],
            "name": state["user_name"],
            "email": state["user_email"]
        })
        
        if result.startswith("ERROR"):
            return {**state, "booking_stage": "ERROR", "error": result}
        
        # SUCCESS: Clear booking state to prevent re-routing loops
        return {
            **state,
            "messages": [AIMessage(content=result)],
            "booking_stage": "CONFIRMED",
            "selected_slot": None,
            "available_slots": None,
            "time_preference": None,
            "asked_for_preference": False,
            "intent": None,
            "error": None
        }
    except Exception as e:
        return {**state, "booking_stage": "ERROR", "error": f"ERROR: {str(e)}"}


def confirm_booking(state: AgentState) -> AgentState:
    """Generate booking confirmation message."""
    # Confirmation is handled in booking_create
    return state


def handle_faq(state: AgentState) -> AgentState:
    """Handle FAQ queries using RAG."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state
    
    try:
        # Retrieve FAQ context
        context = retrieve_faq.invoke({"query": last_user_msg.content})
        
        # Generate response with context
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        system_prompt = f"""You are a helpful assistant for Acme Dental clinic.

Use the following knowledge base information to answer the user's question:

{context}

Provide a clear, concise, and answer based on the knowledge base.
If the information isn't in the knowledge base, politely say you don't have that information."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_user_msg.content)
        ])
        
        return {**state, "messages": [AIMessage(content=response.content)]}
    except Exception as e:
        return {**state, "error": f"ERROR: {str(e)}"}


def respond_to_user(state: AgentState) -> AgentState:
    """Final response node for general conversation."""
    last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return state
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    system_prompt = """You are a concise dental receptionist for Acme Dental.

Respond in 1-2 sentences MAX. Be friendly but brief.
If they're just greeting, greet back and ask: "Would you like to book an appointment or have questions?"
Do NOT make up information. Do NOT offer services we don't have."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_msg.content)
    ])
    
    return {**state, "messages": [AIMessage(content=response.content)]}


def tool_error_handler(state: AgentState) -> AgentState:
    """Handle errors from tool calls."""
    error = state.get("error", "Unknown error occurred")
    
    msg = f"I apologize, but I encountered an issue: {error}\n\nWould you like to try again or can I help you with something else?"
    
    return {**state, "messages": [AIMessage(content=msg)], "error": None}


def route_from_entry(state: AgentState) -> Literal["router", "booking_collect_identity"]:
    """Entry point routing based on intent."""
    intent = state.get("intent")
    booking_stage = state.get("booking_stage", "IDLE")
    
    # If we just confirmed a booking, reset to IDLE and go through router
    if booking_stage == "CONFIRMED":
        state["booking_stage"] = "IDLE"
        return "router"
    
    # If already in booking flow (has intent=BOOK), skip router
    if intent == "BOOK":
        return "booking_collect_identity"
    
    # First time or general query - classify intent
    return "router"


def route_after_router(state: AgentState) -> Literal["handle_faq", "booking_collect_identity", "respond_to_user"]:
    """Route based on classified intent."""
    intent = state.get("intent", "GENERAL")
    
    if intent == "BOOK":
        return "booking_collect_identity"
    elif intent == "FAQ":
        return "handle_faq"
    else:
        return "respond_to_user"


def route_after_identity_check(state: AgentState) -> Literal["ask_for_name_email", "ask_for_time_preference", "parse_time_preference", "booking_check_availability", "parse_slot_selection", "booking_create"]:
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


def route_after_slot_selection(state: AgentState) -> Literal["booking_create", "__end__"]:
    """Route after slot selection attempt."""
    if state.get("selected_slot"):
        return "booking_create"
    # If no valid selection, END (error message already sent)
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
    
    workflow.add_edge("handle_faq", END)
    workflow.add_edge("respond_to_user", END)
    workflow.add_edge("tool_error_handler", END)
    
    return workflow.compile()
