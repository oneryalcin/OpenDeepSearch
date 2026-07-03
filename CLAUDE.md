# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenDeepSearch is a lightweight yet powerful search tool designed for seamless integration with AI agents. It enables deep web search and retrieval, optimized for use with Hugging Face's SmolAgents ecosystem. The tool performs on par with closed source search alternatives on single-hop queries and excels at multi-hop queries.

## Environment Setup

### Installation

```bash
# Install dependencies using pip
pip install -e .
pip install -r requirements.txt

# Alternative: Install with uv (faster package manager)
uv pip install -e .
uv pip install -r requirements.txt

# Alternative: Install with PDM
pdm install
eval "$(pdm venv activate)"
```

### Required API Keys

Before using OpenDeepSearch, you need to set up the following API keys as environment variables:

```bash
# Search provider (choose one)
# Option 1: Serper.dev (default)
export SERPER_API_KEY='your-serper-api-key'

# Option 2: SearXNG
export SEARXNG_INSTANCE_URL='https://your-searxng-instance.com'
export SEARXNG_API_KEY='your-api-key-here'  # Optional

# Reranker (choose one)
# Option 1: Jina AI
export JINA_API_KEY='your-jina-api-key'

# Option 2: Google Cloud Vertex AI
export VERTEXAI_SA_CREDS='/path/to/your/service-account-credentials.json'

# Option 3: Gemini Embeddings
export GOOGLE_API_KEY='your-google-api-key'

# LLM Provider (choose one)
# You can use any provider supported by LiteLLM
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
# Or for custom endpoints
export OPENAI_BASE_URL='https://your-custom-openai-endpoint.com'

# LLM Model selection
export LITELLM_MODEL_ID='openrouter/google/gemini-2.0-flash-001'
export LITELLM_SEARCH_MODEL_ID='openrouter/google/gemini-2.0-flash-001'
```

## Development Commands

### Running the Demo

```bash
# Run the Gradio web interface with default settings (Serper + Jina)
python gradio_demo.py

# With custom parameters
python gradio_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "jina" --search-provider "serper"

# Using SearXNG instead of Serper
python gradio_demo.py --search-provider "searxng" --searxng-instance "https://your-searxng-instance.com"
```

### Running Tests

```bash
# Run basic test
python test_ods.py

# Run evaluation tests
python evals/eval_tasks.py
python evals/eval_gpt_web.py
```

### Linting and Formatting

```bash
# Install dependencies
pip install -e ".[dev]"     # Development dependencies
pip install -e ".[google]"  # Google reranker dependencies
pip install -e ".[all]"     # All dependencies

# Run Black for code formatting
black .

# Run Ruff for linting
ruff check .
```

## Architecture

OpenDeepSearch has the following key components:

1. **API Layer**:
   - `OpenDeepSearchTool`: Main entry point, implements SmolAgents-compatible Tool interface
   - `OpenDeepSearchAgent`: Core search agent that provides the actual functionality

2. **Search Providers**:
   - `SerperAPI`: Interface with Serper.dev search API
   - `SearXNGAPI`: Interface with SearXNG self-hosted search instances

3. **Content Processing Pipeline**:
   - `WebScraper`: Uses Crawl4AI to fetch and extract content from web pages
   - `SourceProcessor`: Handles processing of search results, including reranking
   - `build_context`: Builds a structured context from the processed search results

4. **Reranking**:
   - `InfinitySemanticSearcher`: Local semantic search using Infinity embeddings
   - `JinaReranker`: Cloud-based reranking using Jina AI
   - `VertexAIRanker`: Cross-encoder reranking using Google Cloud Vertex AI
   - `GeminiEmbeddingReranker`: Bi-encoder embedding reranking using Google Gemini embeddings (with fallback)

5. **Integration**:
   - SmolAgents compatibility for use with CodeAgent, ToolCallingAgent, etc.
   - LiteLLM integration for flexible LLM provider selection

## Usage Modes

1. **Default Mode**: Quick and efficient search with minimal latency, ideal for simple queries.
2. **Pro Mode**: More in-depth search with comprehensive web scraping and semantic reranking, better for complex queries but slower.

## Common Workflows

1. **Basic standalone search query**:
   ```python
   from opendeepsearch import OpenDeepSearchTool
   import os
   
   # Set your API keys as environment variables
   os.environ["SERPER_API_KEY"] = "your-serper-api-key"
   os.environ["JINA_API_KEY"] = "your-jina-api-key"
   
   # Create the search tool
   search_agent = OpenDeepSearchTool(
       model_name="gemini/gemini-2.0-flash-001",
       reranker="jina"
   )
   
   # Important: You must call setup() before using forward()
   # when using the tool directly (not through an agent)
   search_agent.setup()
   
   # Now you can call forward
   result = search_agent.forward("Fastest land animal?")
   print(result)
   ```

2. **Integration with SmolAgents' CodeAgent**:
   ```python
   from opendeepsearch import OpenDeepSearchTool
   from smolagents import CodeAgent, LiteLLMModel
   import os
   
   # Set your API keys as environment variables
   os.environ["SERPER_API_KEY"] = "your-serper-api-key"
   os.environ["JINA_API_KEY"] = "your-jina-api-key"
   
   # Create the search tool
   search_tool = OpenDeepSearchTool(
       model_name="gemini/gemini-2.0-flash-001",
       reranker="jina"
   )
   
   # Note: No need to call search_tool.setup() here
   # CodeAgent will automatically call setup() on all tools
   
   # Create the code agent with the search tool
   code_agent = CodeAgent(
       tools=[search_tool],
       model=LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite"),
       additional_authorized_imports=["numpy", "pandas", "matplotlib"],
       max_steps=25,
   )
   
   # Run the agent with a query
   result = code_agent.run("How long would a cheetah at full speed take to run the length of Pont Alexandre III?")
   print(result)
   ```

3. **ReAct agent with multiple tools**:
   ```python
   from opendeepsearch import OpenDeepSearchTool
   from opendeepsearch.wolfram_tool import WolframAlphaTool
   from opendeepsearch.prompts import REACT_PROMPT
   from smolagents import LiteLLMModel, ToolCallingAgent
   import os
   
   # Set your API keys as environment variables
   os.environ["SERPER_API_KEY"] = "your-serper-api-key"
   os.environ["VERTEXAI_SA_CREDS"] = "/path/to/service-account.json"
   os.environ["GOOGLE_API_KEY"] = "your-google-api-key"  # For Gemini fallback
   os.environ["WOLFRAM_ALPHA_APP_ID"] = "your-wolfram-alpha-app-id"
   
   # Create the models and tools
   model = LiteLLMModel("fireworks_ai/llama-v3p1-70b-instruct", temperature=0.7)
   
   # Example using Vertex AI reranker
   search_agent = OpenDeepSearchTool(
       model_name="fireworks_ai/llama-v3p1-70b-instruct",
       reranker="vertex",
       vertex_credentials_path="/path/to/service-account.json"
   )
   
   # Or use Gemini embeddings with fallback
   # search_agent = OpenDeepSearchTool(
   #     model_name="fireworks_ai/llama-v3p1-70b-instruct",
   #     reranker="gemini",
   #     vertex_credentials_path="/path/to/service-account.json",
   #     google_api_key="your-google-api-key"
   # )
   
   wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])
   
   # Note: No need to call search_agent.setup() here
   # ToolCallingAgent will automatically call setup() on all tools
   
   # Create the ReAct agent with both tools
   react_agent = ToolCallingAgent(
       tools=[search_agent, wolfram_tool],
       model=model,
       prompt_templates=REACT_PROMPT
   )
   
   # Run the agent with a query
   result = react_agent.run("What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice")
   print(result)
   ```

## Usage Notes and Common Issues

1. **Initialization Process**:
   - When using OpenDeepSearchTool directly (not through an agent), you must call `.setup()` before calling `.forward()`. 
   - When using tools with agents like CodeAgent or ToolCallingAgent, the agent automatically calls setup() on the tools.
   - Example error if setup() isn't called: `AttributeError: 'OpenDeepSearchTool' object has no attribute 'search_tool'`

2. **API Keys**:
   - All required API keys must be set as environment variables before initializing tools.
   - For quick testing, you can set them directly in your script, e.g., `os.environ["SERPER_API_KEY"] = "your-key"`.

3. **LiteLLM Model Names**:
   - Different providers have different model name formats:
     - Gemini: `gemini/gemini-2.0-flash-001`
     - OpenAI: `openai/gpt-4.1-mini`
     - Anthropic: `anthropic/claude-3-opus-20240229`

4. **Agent Execution Flow**:
   - When using CodeAgent, the execution happens in steps, with each step performing an action (like searching) and analyzing the results.
   - For complex questions, the agent often performs multiple searches and calculations to arrive at an answer.