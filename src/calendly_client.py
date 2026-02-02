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
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_current_user(self) -> dict[str, Any]:
        """Get the current user's information."""
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/users/me",
                headers=self.headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    
    def get_event_types(self, user_uri: str | None = None, use_env: bool = True) -> list[dict[str, Any]]:
        """
        Retrieve event types for the current user.
        
        Args:
            user_uri: Optional user URI. If not provided, fetches for current user.
            use_env: If True, check CALENDLY_EVENT_TYPE_URI env var first.
            
        Returns:
            List of event type objects.
        """
        # Check for cached event type URI in env
        if use_env:
            env_uri = os.getenv("CALENDLY_EVENT_TYPE_URI")
            if env_uri:
                return [{"uri": env_uri, "name": "Dental Check-up"}]
        
        if not user_uri:
            user_info = self.get_current_user()
            user_uri = user_info["resource"]["uri"]
        
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/event_types",
                headers=self.headers,
                params={"user": user_uri},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("collection", [])
    
    def get_available_times(
        self,
        event_type_uri: str,
        start_date: str | None = None,
        end_date: str | None = None
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
        # Calendly API requires ISO datetime format with timezone (e.g., "2026-02-01T00:00:00Z")
        if not start_date:
            start_dt = datetime.utcnow()
            start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif "T" not in start_date:
            # If only date provided, append time
            start_date = f"{start_date}T00:00:00Z"
        
        # CRITICAL: Clamp start_time to ensure it's in the future
        # Calendly requires start_time to be meaningfully in the future, not just > now
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        now_dt = datetime.utcnow().replace(tzinfo=start_dt.tzinfo)
        
        if start_dt < now_dt + timedelta(minutes=5):
            # Reset to now + 5 minutes (Calendly needs a future buffer)
            start_dt = now_dt + timedelta(minutes=5)
            start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"   Adjusted start_time to {start_date} (5min buffer)")
            
        if not end_date:
            # Parse start_date and add 7 days
            end_dt = start_dt + timedelta(days=7)
            end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif "T" not in end_date:
            # If only date provided, append time
            end_date = f"{end_date}T23:59:59Z"
        
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/event_type_available_times",
                headers=self.headers,
                params={
                    "event_type": event_type_uri,
                    "start_time": start_date,
                    "end_time": end_date
                    # Note: "status" parameter removed - not supported by this endpoint
                },
                timeout=10.0
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
        location_location: str | None = None
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
        # Split name into first and last
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
                "timezone": timezone
            }
        }
        
        # Add location if provided
        if location_kind:
            location_obj: dict[str, Any] = {"kind": location_kind}
            if location_location:
                location_obj["location"] = location_location
            payload["location"] = location_obj
        
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/invitees",
                headers=self.headers,
                json=payload,
                timeout=15.0
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("resource", {})
