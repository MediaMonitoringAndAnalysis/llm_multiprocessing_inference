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

### Example 1: Structured inference with textual entries with OpenAI

```python
from llm_multiprocessing_inference import get_answers

structured_system_prompt = "You are a helpful assistant, answer the user's question in a JSON format with the following keys: 'answer', 'confidence', 'source'."
# Define your prompts
prompts = [
    [
        {"role": "system", "content": structured_system_prompt},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": structured_system_prompt},
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
]

# Get answers
answers = get_answers(
    prompts=prompts,
    default_response={},
    response_type="structured",
    api_pipeline="OpenAI",  # or "Perplexity" or "Ollama"
    api_key="your-api-key",  # only needed for OpenAI or Perplexity
    model="your-model-here",
)
```

### Example 2: Unstructured inference with textual entries with Ollama

```python
from llm_multiprocessing_inference import get_answers

unstructured_system_prompt = (
    "You are a helpful assistant, answer the user's question in a free text format."
)
# Define your prompts
prompts = [
    [
        {"role": "system", "content": unstructured_system_prompt},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": unstructured_system_prompt},
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
]

# Get answers
answers = get_answers(
    prompts=prompts,
    default_response="-",
    response_type="unstructured",
    api_pipeline="Ollama",  # or "Perplexity" or "OpenAI"
    model="phi4-mini:3.8b-q4_K_M"
)
```

### Example 3: Structured inference with images with OpenAI

```python
import base64
from llm_multiprocessing_inference import get_answers


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


structured_system_prompt = "You are a helpful assistant, answer the user's question in a JSON format with the following keys: 'answer', 'confidence', 'source'."

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

# Define your prompts
prompts = [
    [
        {"role": "system", "content": structured_system_prompt},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": structured_system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
]

answers = get_answers(
    prompts=prompts,
    default_response={},
    response_type="structured",
    api_pipeline="OpenAI",
    api_key="your-api-key",
    model="gpt-4o-mini",
)
```

### Example 4: Structured inference with images with Ollama

```python
from llm_multiprocessing_inference import get_answers

structured_system_prompt = 'From this image, return a JSON with the asked information. Do not return any false information. Do not return anything other than the JSON dictionary. If you cannot find one information or are unsure, return "-".'

# Path to your image
image_path = "path_to_your_image.jpg"

# Define your prompts
prompts = [
    [
        {"role": "system", "content": structured_system_prompt},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": structured_system_prompt},
        {
            "role": "user",
            "content": 'From this image, return a JSON dictionary with the following information about the image metadata: "title", "description", "date", "source", "url".',
            "images": [image_path],
        },
    ],
]

answers = get_answers(
    prompts=prompts,
    default_response={},
    response_type="structured",
    api_pipeline="Ollama",
    model="llava:7b-v1.6-mistral-q4_K_M",
)
```




## Tree Structure
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
