"""Main entry point for the Acme Dental AI Agent."""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import create_acme_dental_agent


def main():
    load_dotenv()
    agent = create_acme_dental_agent()

    print("🦷 Acme Dental AI Agent")
    print("=" * 50)
    print("Welcome! I can help you book appointments or answer questions about our clinic.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    # Initialize persistent state for the conversation
    state = {
        "messages": [],
        "intent": None,
        "flow": "IDLE",
        "user_name": None,
        "user_email": None,
        "time_preference": None,
        "asked_for_preference": False,
        "selected_slot": None,
        "available_slots": None,
        "error": None,
        "lookup_email": None,
        "matched_events": None,
        "selected_event_uri": None,
        "confirmed": False,
    }

    last_printed_count = 0  # Track how many messages we've printed

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nThank you for contacting Acme Dental! Have a great day! 🦷")
            break

        try:
            # Append new user message to existing conversation
            state["messages"].append(HumanMessage(content=user_input))

            # Invoke the graph with the current state
            result = agent.invoke(state)

            # Update state with results, preserving collected information
            state["messages"] = result.get("messages", state["messages"])
            state["intent"] = result.get("intent", state["intent"])

            # Preserve user info and booking data once collected
            if result.get("user_name"):
                state["user_name"] = result["user_name"]
            if result.get("user_email"):
                state["user_email"] = result["user_email"]
            if result.get("time_preference"):
                state["time_preference"] = result["time_preference"]
            if result.get("asked_for_preference") is not None:
                state["asked_for_preference"] = result["asked_for_preference"]
            if result.get("selected_slot"):
                state["selected_slot"] = result["selected_slot"]
            if result.get("available_slots"):
                state["available_slots"] = result["available_slots"]

            # Persist unified flow state
            state["flow"] = result.get("flow", "IDLE")

            if result.get("lookup_email"):
                state["lookup_email"] = result["lookup_email"]
            if result.get("matched_events"):
                state["matched_events"] = result["matched_events"]
            if result.get("selected_event_uri"):
                state["selected_event_uri"] = result["selected_event_uri"]
            if result.get("confirmed") is not None:
                state["confirmed"] = result["confirmed"]

            state["error"] = result.get("error")

            # Print only NEW AI messages (delta)
            messages = state["messages"]
            new_messages = messages[last_printed_count:]

            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    print(f"\nAgent: {msg.content}\n")

            last_printed_count = len(messages)

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
