"""AI Agent for the Acme Dental Clinic."""

from src.graph import create_booking_graph


def create_acme_dental_agent():
    """Create and return the compiled booking agent graph."""
    return create_booking_graph()
