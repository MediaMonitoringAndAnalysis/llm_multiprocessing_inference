# OpenAI Multiprocess Inference

A Python package for efficient parallel inference using OpenAI and other LLM APIs (can only be used on .py scripts and not on jupyter notebooks).

## Prerequisites

- Python 3.10+
- Ollama installed: [text](https://ollama.com/download)

## Installation

```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/llm_multiprocessing_inference.git
```

## Features

- 🚀 Parallel inference using asyncio
- 🔄 Support for multiple LLM APIs (OpenAI, Perplexity)
- 📊 Progress bar for bulk operations
- 🎯 Structured and unstructured response handling
- 🛡️ Built-in error handling and retries
- ⚡ Optimized for high-throughput scenarios
- 🔄 Support for Ollama

## Quick Start
```python
from llm_multiprocessing_inference import get_answers

# Define your prompts
prompts = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "What is the capital of Germany?"}],
]

# Get answers
answers = get_answers(
prompts=prompts,
default_response="{}",
response_type="structured",
api_pipeline="OpenAI",
api_key="your-api-key"
)
```

Here's the complete tree structure for the package:
```plaintext
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
└── src/
    └── openai_multiproc_inference/
        ├── __init__.py
        └── inference.py
```