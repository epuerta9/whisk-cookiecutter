# {{ cookiecutter.project_name }}

A Whisk client application using {{ cookiecutter.framework }}.

## Features

{% if cookiecutter.framework == 'llama-index' %}
- LlamaIndex-based RAG implementation
- ChromaDB vector store integration
- Token counting and usage tracking
{% endif %}

{% if cookiecutter.framework == 'langchain' %}
- Langchain-based RAG implementation
- ChromaDB vector store with OpenAI embeddings
- Flexible chain composition
{% endif %}

{% if cookiecutter.framework == 'crewai' %}
- Multi-agent system using CrewAI
- Collaborative task solving
- Role-based agent interactions
{% endif %}

{% if cookiecutter.framework == 'smol-agents' %}
- Lightweight agent implementation
- Minimal dependencies
- Focused task execution
{% endif %} 