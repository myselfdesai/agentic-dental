# Developer Guide: Acme Dental AI Agent

This guide covers how to set up, configure, and run the agent locally.

## Prerequisities

- Python 3.11+
- `uv` (recommended for dependency management) or `pip`
- A Calendly account (Premium required for API access)
- OpenAI or Anthropic API Key

## Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   # OR
   pip install -e .
   ```

2. **Environment Variables**:
   Copy the example file:
   ```bash
   cp .env.example .env
   ```
   
   Fill in the required keys:

   | Variable | Description | Required? |
   |----------|-------------|-----------|
   | `OPENAI_API_KEY` | For GPT-4o-mini logic | Yes (if `LLM_PROVIDER=openai`) |
   | `ANTHROPIC_API_KEY` | For Claude 3 logic | Yes (if `LLM_PROVIDER=anthropic`) |
   | `CALENDLY_API_TOKEN` | Personal Access Token from Calendly Integrations | **Yes** |
   | `LANGCHAIN_TRACING_V2` | Set to `true` for LangSmith tracing | No (Recommended) |
   | `LANGCHAIN_API_KEY` | LangSmith API Key | No |

3. **LLM Provider Configuration**:
   By default, the agent uses OpenAI. To switch to Anthropic:
   ```bash
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-...
   LLM_MODEL=claude-3-haiku-20240307 # Optional override
   ```

## Running the Agent

Start the CLI interface:
```bash
make run
# OR
uv run python src/main.py
```

## Running Tests

Run the LangSmith evaluation script (ensure env vars are set):
```bash
# Note: Makefile test target runs pytest, not eval_router.py yet.
# You can run eval script manually:
uv run python tests/eval_router.py
```

## Linting

Use `make` to check for style issues:
```bash
make lint   # Lint code
make format # Format code
make check  # run both
```
