# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Setup

1. Install dependencies:
```bash
pip install -e '.[dev]'
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run the app:
```bash
whisk run app.main:kitchen
```

## Project Structure

```
app/
├── main.py           # Main app with kitchen object
├── handlers/         # Query and storage handlers
├── dependencies/     # LLM and vector store setup
└── utils/           # Utilities like token counting
```

## Usage

### Query Handler
```python
response = await client.query({
    "query": "What is...",
    "metadata": {"filter": "value"}
})
```

### Storage Handler
```python
response = await client.store({
    "id": 1,
    "name": "document.pdf",
    "data": b"..."
})
```

## Development

1. Install dev dependencies:
```bash
pip install -e '.[dev]'
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
``` 