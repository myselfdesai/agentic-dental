# Graph Design Doc: Acme Dental Booking Agent

This document outlines the architecture, state management, and flow logic of the AI booking agent.

## 1. Overview
The agent uses a **LangGraph** state machine to handle three primary flows:
- **BOOK**: Create new appointments.
- **CANCEL**: Cancel existing appointments.
- **RESCHEDULE**: Cancel an existing appointment and book a new one.

It supports **context switching** (interrupts) allowing users to change intent mid-stream (e.g., switching from Reschedule to Book).

## 2. State Schema (`AgentState`)

The state is a typed dictionary (`TypedDict`) shared across all nodes.

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]   # Chat history
    
    # Control Flow
    intent: str | None            # DETECTED intent (BOOK, CANCEL, RESCHEDULE)
    flow: Literal["IDLE", "BOOK", "CANCEL", "RESCHEDULE"] # ACTIVE flow
    
    # User Identity (Global)
    user_name: str | None
    user_email: str | None
    
    # Booking data
    time_preference: str | None   # Natural language (e.g. "tuesday morning")
    asked_for_preference: bool    # Tracking flag
    available_slots: list[dict] | None # Fetched from Calendly
    selected_slot: str | None     # ISO 8601 string
    
    # Event data (Cancel/Reschedule)
    lookup_email: str | None      # Email used for lookup
    matched_events: list[dict] | None # Found appointments
    selected_event_uri: str | None    # URI of appointment to act on
    confirmed: bool               # Final confirmation flag
    
    error: str | None             # Error messaging
```

## 3. Intents & Stages

- **IDLE**: Agent accepts any command.
- **BOOK**: focused on collecting slots, checking availability, and booking.
- **CANCEL**: focused on looking up event and confirming deletion.
- **RESCHEDULE**: Hybrid flow.
  1. Lookup existing event.
  2. Select event.
  3. Transition to Booking logic (collect time).
  4. Create new booking -> Cancel old booking.

## 4. Nodes List

| Node Name | Description | input | output (next) |
|-----------|-------------|-------|---------------|
| `check_existing_flow` | Entry point. Checking for **interrupts** (e.g. "cancel" while booking). | `messages` | `route_from_entry` |
| `router` | Classifies intent for new conversations. | `messages` | `route_after_router` |
| `booking_collect_identity` | Extracts Name/Email. | `messages` | `route_after_identity_check` |
| `ask_for_name_email` | Prompts user for missing info. | `state` | `END` |
| `lookup_events` | Finds appointments for Cancel/Reschedule. | `user_email` | `route_after_lookup` |
| `select_event` | Handles user selection (1, 2, Yes/No). | `messages` | `route_after_selection` |
| `ask_for_time_preference` | Asks "When would you like to come in?". | `state` | `END` |
| `parse_time_preference` | Extracts "Tuesday morning" from text. | `messages` | `booking_check_availability` |
| `booking_check_availability` | Calls Calendly to find slots. | `time_preference` | `route_after_availability_check` |
| `parse_slot_selection` | Handles "Use slot 1" or "actually wednesday". | `messages` | `route_after_slot_selection` |
| `booking_create` | Creates booking via API (and cancels old if Reschedule). | `selected_slot` | `confirm_booking` |
| `confirm_action` | Cancels appointment (Cancel flow only). | `selected_event` | `END` |

## 5. Flow Interrupt Logic

The `check_existing_flow` node runs **first** on every turn.
- If `flow` is IDLE, we proceed to `router`.
- If `flow` is active (e.g. BOOK), it checks the **latest user message**.
  - If it contains high-priority keywords ("cancel", "book new", "reschedule"), it **overrides** the current `flow` and `intent`.
  - This prevents users from getting trapped in loops.

## 6. Failure Cases & Handling

1.  **No Slots Found**:
    - Trigger: `booking_check_availability` returns empty list.
    - Behavior: Defaults to showing **all** available slots for the next 7 days.

2.  **Invalid Email / No Bookings**:
    - Trigger: `lookup_events` finds 0 events.
    - Behavior: Message "I couldn't find...", asks if they want to try another email or book new.

3.  **Ambiguous Selection**:
    - Trigger: User types "change time" instead of "1".
    - Behavior: `parse_slot_selection` clears slots and routes back to `parse_time_preference` to understand the new request.

## 7. Example Conversations

### Happy Path: Reschedule
```
User: Reschedule
Agent: Email?
User: user@example.com
Agent: Found appt Feb 3rd 10am. Is this it?
User: Yes
Agent: When do you want to move it to?
User: Next wednesday
Agent: Slots: 1. Wed 10am, 2. Wed 11am.
User: 1
Agent: Booked Wed 10am. Cancelled Feb 3rd.
```

### Interrupt Path
```
User: Cancel
Agent: Email?
User: Actually I want to book a new one
Agent: (Switches to BOOK flow) Do you have a preferred time?
User: Friday
...
```
