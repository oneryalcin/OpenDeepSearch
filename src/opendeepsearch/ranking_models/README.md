# Reranking Models in OpenDeepSearch

This directory contains the reranking models used by OpenDeepSearch to improve search relevance by reordering search results based on their semantic similarity to the query.

## Available Rerankers

OpenDeepSearch supports multiple reranking options to suit different needs:

### 1. Infinity Reranker (Default)

- **Type**: Bi-encoder (embeddings-based)
- **Model**: Uses open-source models like `Alibaba-NLP/gte-Qwen2-7B-instruct`
- **Dependencies**: Runs locally or connects to a local Infinity Embedding API server
- **Advantages**: No additional API keys required, works offline
- **Best for**: Development and testing, privacy-sensitive applications

### 2. Jina Reranker

- **Type**: Bi-encoder (embeddings-based)
- **Model**: Uses Jina's powerful embedding models like `jina-embeddings-v3`
- **Dependencies**: Requires `JINA_API_KEY` environment variable
- **Advantages**: High-quality results, simple integration
- **Best for**: Production use with moderate query volume

### 3. Vertex AI Reranker

- **Type**: Cross-encoder (direct relevance scoring)
- **Model**: Uses Google's `semantic-ranker-default-004`
- **Dependencies**: Requires Google Cloud service account credentials (`VERTEXAI_SA_CREDS`)
- **Advantages**: High accuracy, evaluates query-document pairs directly
- **Best for**: Enterprise applications requiring high-quality reranking

### 4. Gemini Embedding Reranker

- **Type**: Bi-encoder (embeddings-based)
- **Model**: Uses Google's state-of-the-art embedding models:
  - `text-embedding-large-exp-03-07` (via Vertex AI)
  - `gemini-embedding-exp-03-07` (via Gemini API as fallback)
- **Dependencies**: 
  - Primary: Google Cloud service account credentials (`VERTEXAI_SA_CREDS`)
  - Fallback: Google API key (`GOOGLE_API_KEY`)
- **Advantages**: High-quality embeddings, fallback mechanism, Matryoshka embedding support
- **Best for**: Production use with high quality requirements

## Usage

To use a specific reranker, specify it when creating the `OpenDeepSearchTool`:

```python
from opendeepsearch import OpenDeepSearchTool

# Using Infinity (default)
search_tool = OpenDeepSearchTool(reranker="infinity")

# Using Jina
search_tool = OpenDeepSearchTool(reranker="jina")

# Using Vertex AI
search_tool = OpenDeepSearchTool(
    reranker="vertex",
    vertex_credentials_path="/path/to/service-account.json",
    vertex_project_id="your-gcp-project-id"
)

# Using Gemini Embeddings (with fallback)
search_tool = OpenDeepSearchTool(
    reranker="gemini",
    vertex_credentials_path="/path/to/service-account.json",
    vertex_project_id="your-gcp-project-id",
    google_api_key="your-google-api-key",
    output_dimensions=768  # Optional, reduce dimensions for efficiency
)
```

## Technical Details

### Bi-encoder vs. Cross-encoder

OpenDeepSearch supports both bi-encoder and cross-encoder reranking approaches:

1. **Bi-encoder** (Infinity, Jina, Gemini):
   - Computes separate embeddings for queries and documents
   - Uses dot product to calculate similarity scores
   - Generally faster but potentially less accurate
   - Better for caching and pre-computing embeddings

2. **Cross-encoder** (Vertex AI):
   - Takes query-document pairs as input
   - Directly computes relevance scores
   - Generally more accurate but slower
   - Cannot pre-compute or cache embeddings

### Installing Dependencies

Each reranker requires specific dependencies:

```bash
# For Jina
pip install opendeepsearch
export JINA_API_KEY='your-jina-api-key'

# For Google rerankers (Vertex AI and/or Gemini)
pip install opendeepsearch[google]
export VERTEXAI_SA_CREDS='/path/to/service-account.json'
export GOOGLE_API_KEY='your-google-api-key'  # For Gemini fallback

# For all dependencies
pip install opendeepsearch[all]
```

## Creating Your Own Reranker

To implement your own reranker, inherit from `BaseSemanticSearcher` and implement the `_get_embeddings()` method:

```python
from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher
import torch
from typing import List

class MyCustomReranker(BaseSemanticSearcher):
    def __init__(self):
        # Initialize your embedding model here
        super().__init__()
        self.model = YourEmbeddingModel()
        
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        # Implement your embedding logic here
        embeddings = self.model.encode(texts)
        return torch.tensor(embeddings)
```

The base class automatically handles:
- Similarity score calculation
- Score normalization (softmax, scaling, or none)
- Document reranking
- Top-k selection

## Using Infinity Rerankers

For high-performance reranking, we support [Infinity](https://github.com/michaelfeil/infinity) rerankers which offer state-of-the-art performance. To use an Infinity reranker, first start the Infinity server:

```bash
# requires ~16-32GB VRAM NVIDIA Compute Capability >= 8.0
docker run \
-v $PWD/data:/app/.cache --gpus "0" -p "7997":"7997" \
michaelf34/infinity:0.0.68-trt-onnx \
v2 --model-id Alibaba-NLP/gte-Qwen2-7B-instruct --revision "refs/pr/38" \
--dtype bfloat16 --batch-size 8 --device cuda --engine torch --port 7997 \
--no-bettertransformer
```

This will start an Infinity server using the Qwen2-7B-instruct model. The server will be available at `localhost:7997`.

Key parameters:
- `--model-id`: The Hugging Face model ID to use
- `--dtype`: Data type for inference (bfloat16 recommended for modern GPUs)
- `--batch-size`: Batch size for inference
- `--port`: Port to expose the server on

## Using Jina AI Rerankers

Jina AI provides powerful embedding models through their API service. The `JinaReranker` class offers a simple way to leverage these models:

```python
from opendeepsearch.ranking_models.jina_reranker import JinaReranker

# Initialize with your API key
reranker = JinaReranker(api_key="your_api_key")  # or set JINA_API_KEY env variable

# Example usage
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a type of machine learning",
    "Natural language processing uses machine learning"
]

# Get reranked documents
reranked_docs = reranker.get_reranked_documents(query, documents, top_k=2)
```

The JinaReranker uses Jina's v3 embeddings by default, which provides:
- 1024-dimensional embeddings
- Optimized for text matching tasks
- State-of-the-art performance for semantic search