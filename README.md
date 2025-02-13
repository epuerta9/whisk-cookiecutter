# Whisk Project Template

A cookiecutter template for creating KitchenAI Whisk projects. This template helps you quickly set up a project with the right structure and dependencies for your AI implementation.

## Features

- Framework selection (llama-index or langchain)
- Use case templates (RAG, Chat, Agent)
- Project structure options (Simple or Modular)
- LlamaIndex Implementation Options:
  - Context Chat: Chat with context from your data
  - Personality Chat: Styled chat interactions
  - ReAct Chat: Reasoning and action agent
  - Agent Pipeline: Complex query workflows
- Automatic dependency management
- NATS integration
- Type-safe dependency injection

## Implementation Options

### Context Chat
Best for question answering over documents:
```python
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=1500)
)
response = chat_engine.chat("What did Paul Graham do?")
```

### Personality Chat
Styled chat interactions with customizable personalities:
```python
chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)
response = chat_engine.chat("Tell me about AI")
```

### ReAct Chat
Complex reasoning with tool use:
```python
chat_engine = index.as_chat_engine(
    chat_mode="react",
    verbose=True
)
response = chat_engine.chat("Use the tool to find...")
```

### Agent Pipeline
Advanced query workflows:
```python
agent = QueryPipelineAgent(
    tools=[tool1, tool2],
    verbose=True
)
response = agent.chat("Complex multi-step task...")
```

## Project Structure Options

### Simple Structure
Best for smaller projects or single-purpose applications:
```
project_name/
├── app.py              # Main application with handlers
├── config.yml          # Configuration
├── pyproject.toml      # Dependencies
└── README.md          # Documentation
```

### Modular Structure
Better for larger projects with multiple components:
```
project_name/
├── apps/              # Sub-apps for different domains
│   ├── chat/         # Chat-related handlers
│   ├── rag/          # RAG-related handlers
│   └── agent/        # Agent-related handlers
├── core/             # Core functionality
│   ├── config.py     # Configuration management
│   ├── deps.py       # Dependency definitions
│   └── utils.py      # Shared utilities
├── app.py            # Main application (mounts sub-apps)
├── config.yml        # Configuration
├── pyproject.toml    # Dependencies
└── README.md         # Documentation
```

## Usage

1. Install cookiecutter:
```bash
pip install cookiecutter
```

2. Create a new project:
```bash
cookiecutter gh:kitchenai/whisk-template
```

3. Answer the prompts about your project:
- Project name
- Framework choice (llama-index/langchain)
- Use case (rag/chat/agent)
- LlamaIndex implementation (if using llama-index)
- Chat personality (if using personality chat)
- Project structure (simple/modular)
- NATS configuration

4. Install dependencies:
```bash
cd your_project_name
pip install -e .
```

## Testing

### Unit Tests
First, install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-mock
```

Run the test suite:
```bash
pytest
```

The tests will verify:
- Handler input/output contracts
- Dependency injection
- Mock LLM responses
- Tool integrations

### LLM Evaluation
For deeper evaluation of LLM outputs, install deepeval:
```bash
pip install deepeval
```

Run the evaluation suite:
```bash
# Set your OpenAI key for evaluations
export OPENAI_API_KEY=your-key-here

# Run evals
pytest eval/test_llm_eval.py
```

The evaluation tests:
- Contextual relevancy (how well responses use provided context)
- Faithfulness (accuracy relative to source material)
- Answer relevancy (response appropriateness for queries)

Example eval output:
```
test_handler_outputs
✓ Context relevancy score: 0.89
✓ Faithfulness score: 0.92
✓ Answer relevancy score: 0.85
```

### Test Configuration
The test suite is configured in pytest.ini:
- Async test support enabled
- Short tracebacks for clarity
- Verbose output
- Automatic test discovery

## Configuration

The project uses a config.yml file for configuration:

```yaml
nats:
  url: "nats://localhost:4222"
  user: "your-user"
  password: "your-password"
client:
  id: "your-client-id"
```

## License

This project is licensed under the terms of the license you choose during project creation.