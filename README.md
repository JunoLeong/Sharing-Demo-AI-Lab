# AI Lab Demo

Hands-on set of demos you can run locally. We start with a simple chatbot, then build a RAG (Retrieval-Augmented Generation) app, and finish with an AI Agent that turns a meeting transcript into clean minutes.

## What’s inside

**Part 1 Simple Chatbot App** : Minimal local LLM chat (Ollama + LangChain)

**Part 2 RAG** : Upload a PDF → embed with Ollama → index in Chroma → ask grounded questions in Streamlit

**Part 3 AI Agent** : Two-agent CrewAI pipeline: Minute Taker (JSON) → Minute Formatter (Markdown)

## Prerequisites
- Python 3.9 - 3.12 (recommended)
- Ollama installed & running locally
  - Ollama installed and running
  - macOS: ```brew install ollama```
  - Windows/Linux: see https://ollama.com

- Models pulled in Ollama:
  
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```
> Note: You can replace llama3.2 with any chat-capable model available in Ollama (e.g., mistral, qwen2.5).

- (Recommended) a fresh virtual environment

