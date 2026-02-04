from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm import get_llm
from src.state import AgentState
from src.tools import retrieve_faq


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
