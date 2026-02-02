"""Calendly API client for scheduling appointments."""

import os
from datetime import datetime, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class CalendlyClient:
    """Client for interacting with Calendly Scheduling API."""

    def __init__(self):
        self.api_key = os.getenv("CALENDLY_API_TOKEN")
        if not self.api_key:
            raise ValueError("CALENDLY_API_TOKEN not found in environment variables")

        self.base_url = "https://api.calendly.com"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def get_current_user(self) -> dict[str, Any]:
        """Get the current user's information."""
        with httpx.Client() as client:
            response = client.get(f"{self.base_url}/users/me", headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()

    def get_event_types(self, user_uri: str | None = None, use_env: bool = True) -> list[dict[str, Any]]:
        """
        Retrieve event types for the current user.

        Args:
            user_uri: Optional specific user URI. If None, fetches for current user.
            use_env: If True, try to use CALENDLY_EVENT_TYPE_URI from env first.
        """
        if not user_uri:
            user_data = self.get_current_user()
            user_uri = user_data.get("resource", {}).get("uri")

        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/event_types", headers=self.headers, params={"user": user_uri}, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("collection", [])

    def get_available_times(
        self, event_type_uri: str, start_date: str | None = None, end_date: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get available time slots for an event type.

        Args:
            event_type_uri: URI of the event type
            start_date: Start date in ISO format (defaults to today)
            end_date: End date in ISO format (defaults to 7 days from start)

        Returns:
            List of available time objects with start_time and invitees_remaining.
        """
        if not start_date:
            start_dt = datetime.utcnow()
            start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif "T" not in start_date:
            start_date = f"{start_date}T00:00:00Z"

        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        now_dt = datetime.utcnow().replace(tzinfo=start_dt.tzinfo)

        if start_dt < now_dt + timedelta(minutes=5):
            start_dt = now_dt + timedelta(minutes=5)
            start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if not end_date:
            end_dt = start_dt + timedelta(days=7)
            end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif "T" not in end_date:
            end_date = f"{end_date}T23:59:59Z"

        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/event_type_available_times",
                headers=self.headers,
                params={"event_type": event_type_uri, "start_time": start_date, "end_time": end_date},
                timeout=10.0,
            )

            response.raise_for_status()
            data = response.json()
            return data.get("collection", [])

    def create_invitee(
        self,
        event_type_uri: str,
        start_time: str,
        name: str,
        email: str,
        timezone: str = "America/New_York",
        location_kind: str | None = None,
        location_location: str | None = None,
    ) -> dict[str, Any]:
        """
        Create an invitee (book an appointment).

        Args:
            event_type_uri: URI of the event type
            start_time: Start time in UTC ISO format (e.g., "2025-10-02T18:30:00Z")
            name: Invitee's full name
            email: Invitee's email address
            timezone: Invitee's timezone (IANA format)
            location_kind: Optional location kind (e.g., "zoom_conference", "google_meet")
            location_location: Optional location details (required for some location kinds)

        Returns:
            Created invitee object with cancel_url, reschedule_url, etc.
        """
        name_parts = name.strip().split(maxsplit=1)
        first_name = name_parts[0] if name_parts else name
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        payload: dict[str, Any] = {
            "event_type": event_type_uri,
            "start_time": start_time,
            "invitee": {
                "name": name,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "timezone": timezone,
            },
        }

        if location_kind:
            location_obj: dict[str, Any] = {"kind": location_kind}
            if location_location:
                location_obj["location"] = location_location
            payload["location"] = location_obj

        with httpx.Client() as client:
            response = client.post(f"{self.base_url}/invitees", headers=self.headers, json=payload, timeout=15.0)

            response.raise_for_status()
            data = response.json()
            return data.get("resource", {})

    def list_scheduled_events(self, invitee_email: str) -> list[dict[str, Any]]:
        """
        List scheduled events for a specific invitee email.

        Args:
            invitee_email: Email address of the invitee

        Returns:
            List of scheduled event objects with invitee details
        """
        user_data = self.get_current_user()
        org_uri = user_data.get("resource", {}).get("current_organization")

        if not org_uri:
            return []

        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/scheduled_events",
                headers=self.headers,
                params={"organization": org_uri, "status": "active", "count": 100},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            events = data.get("collection", [])

        matching_events = []
        for event in events:
            event_uri = event.get("uri")

            with httpx.Client() as client:
                inv_response = client.get(f"{event_uri}/invitees", headers=self.headers, timeout=10.0)
                inv_response.raise_for_status()
                inv_data = inv_response.json()
                invitees = inv_data.get("collection", [])

            for invitee in invitees:
                if invitee.get("email", "").lower() == invitee_email.lower():
                    event["invitee"] = invitee
                    matching_events.append(event)
                    break

        return matching_events

    def cancel_event(self, event_uri: str, reason: str = "Canceled by user request") -> dict[str, Any]:
        """
        Cancel a scheduled event.

        Args:
            event_uri: URI of the event to cancel
            reason: Reason for cancellation

        Returns:
            Cancellation response data
        """
        with httpx.Client() as client:
            response = client.post(
                f"{event_uri}/cancellation", headers=self.headers, json={"reason": reason}, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("resource", {})
