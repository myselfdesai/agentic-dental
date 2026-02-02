import asyncio
import os

# Ensure we can import src
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph import router

load_dotenv()

# 1. Define Dataset
examples = [
    ("I want to book an appointment", "BOOK"),
    ("Cancel my booking please", "CANCEL"),
    ("I need to reschedule for next week", "RESCHEDULE"),
    ("What are your opening hours?", "GENERAL"),  # Router usually falls back to GENERAL/FAQ check downstream?
    # Actually router maps to "GENERAL" or "FAQ".
    # We'll expect GENERAL or FAQ depending on LLM.
    # Let's say GENERAL for now.
    ("Can I schedule a checkup", "BOOK"),
]


# 2. Define Evaluator
def exact_match(run, example):
    # run.outputs is the state returned by router
    predicted = run.outputs.get("intent")
    reference = example.outputs.get("intent")
    return {"score": 1 if predicted == reference else 0}


# 3. Define Target Function
def target(inputs):
    """Wrap the router node."""
    msg = inputs["text"]
    state = {"messages": [HumanMessage(content=msg)], "flow": "IDLE"}
    # Call the router function directly
    result = router(state)
    return {"intent": result.get("intent")}


async def run_eval():
    client = Client()

    # Create dataset in LangSmith programmatically (or assume it exists)
    dataset_name = "Dental Router Smoke Test"

    # Check if dataset exists, if not create it
    ds = None
    if client.has_dataset(dataset_name=dataset_name):
        ds = client.read_dataset(dataset_name=dataset_name)
    else:
        ds = client.create_dataset(dataset_name=dataset_name)
        for text, label in examples:
            client.create_example(inputs={"text": text}, outputs={"intent": label}, dataset_id=ds.id)

    # Run evaluation
    results = evaluate(target, data=dataset_name, evaluators=[exact_match], experiment_prefix="router-smoke-test")

    print("\nResults:", results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_eval())
