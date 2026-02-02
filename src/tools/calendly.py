"""LangChain tools for Calendly integration."""

from datetime import datetime
from typing import Annotated

from langchain_core.tools import tool

from src.calendly_client import CalendlyClient


@tool
def get_calendly_event_type() -> str:
    """Get the URI of the dental check-up event type from Calendly."""
    try:
        client = CalendlyClient()
        event_types = client.get_event_types()

        if not event_types:
            return "ERROR: No event types found in Calendly account."

        # Return the first event type (assuming single dental check-up service)
        event_type = event_types[0]
        uri = event_type.get("uri", "")
        name = event_type.get("name", "Unknown")

        # DEBUG: Print event type info
        print("\n🔧 DEBUG - Event Type Found:")
        print(f"   Name: {name}")
        print(f"   URI: {uri}")
        print(f"   Add to .env: CALENDLY_EVENT_TYPE_URI={uri}\n")

        return uri
    except Exception as e:
        return f"ERROR: Failed to fetch event type: {str(e)}"


@tool
def check_availability(days_ahead: Annotated[int, "Number of days from today to check availability"] = 7) -> str:
    """
    Check available appointment slots for the next N days.

    Returns a formatted string with available time slots.
    """
    try:
        client = CalendlyClient()

        # Get event type
        event_types = client.get_event_types()
        if not event_types:
            return "ERROR: No event types configured."

        event_type_uri = event_types[0]["uri"]

        # Print debug info
        print("\n🔧 DEBUG - Checking availability...")
        print(f"   Event Type URI: {event_type_uri}")

        # IMPORTANT: Pass None to use current time (not midnight)
        # The client will default to utcnow() which prevents past-time errors
        slots = client.get_available_times(event_type_uri, None, None)

        if not slots:
            return f"No available slots found for the next {days_ahead} days."

        # Format slots (limit to first 10 for readability)
        formatted_slots = []
        for i, slot in enumerate(slots[:10], 1):
            start_time = slot.get("start_time", "")
            # Parse and format time nicely
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%A, %B %d at %I:%M %p UTC")
                formatted_slots.append(f"{i}. {formatted_time} ({start_time})")
            except Exception:
                formatted_slots.append(f"{i}. {start_time}")

        print(f"   Found {len(slots)} slots\n")
        return "\n".join(formatted_slots)
    except Exception as e:
        print(f"   ERROR: {str(e)}\n")
        return f"ERROR: Failed to check availability: {str(e)}"


@tool
def create_booking(
    start_time: Annotated[str, "Start time in UTC ISO format (e.g., 2025-10-02T18:30:00Z)"],
    name: Annotated[str, "Patient's full name"],
    email: Annotated[str, "Patient's email address"],
    timezone: Annotated[str, "Patient's timezone in IANA format"] = "America/New_York",
) -> str:
    """
    Create a dental check-up appointment booking.

    Args:
        start_time: Appointment start time in UTC ISO format
        name: Patient's full name
        email: Patient's email address
        timezone: Patient's timezone (default: America/New_York)

    Returns:
        Confirmation message with booking details.
    """
    try:
        client = CalendlyClient()

        # Get event type AND its location configuration
        event_types = client.get_event_types()
        if not event_types:
            return "ERROR: No event types configured."

        event_type = event_types[0]
        event_type_uri = event_type["uri"]

        # Extract location from event type config (REQUIRED by Calendly API)
        location_kind = None
        location_location = None
        if "locations" in event_type and event_type["locations"]:
            first_location = event_type["locations"][0]
            location_kind = first_location.get("kind")
            location_location = first_location.get("location")

        # Create invitee with location
        _ = client.create_invitee(
            event_type_uri=event_type_uri,
            start_time=start_time,
            name=name,
            email=email,
            timezone=timezone,
            location_kind=location_kind,
            location_location=location_location,
        )

        # Format confirmation
        # cancel_url = result.get("cancel_url", "N/A")
        # reschedule_url = result.get("reschedule_url", "N/A")

        return f"""✅ All set! Your appointment is confirmed.

📅 **{name}** - {email}
🕐 {start_time} ({timezone})

You'll receive a confirmation email shortly. You can reschedule or cancel anytime using the links in the email.

Have a great day! 🦷"""
    except Exception as e:
        return f"ERROR: Failed to create booking: {str(e)}"
