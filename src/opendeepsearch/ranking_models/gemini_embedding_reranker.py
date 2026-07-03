import os
import json
import torch
import traceback
import importlib.util
from typing import List, Optional
from dotenv import load_dotenv

# Vertex AI imports
from google.oauth2 import service_account

# Gemini API import
from google import genai
from google.genai.types import EmbedContentConfig

# Note: We'll import vertexai and related modules dynamically
# in the _get_vertex_ai_embeddings method to handle import errors gracefully

from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher

class GeminiEmbeddingReranker(BaseSemanticSearcher):
    """
    Semantic searcher implementation using Google's Gemini embedding models.
    
    This reranker has two modes of operation:
    1. Primary: Use Vertex AI's embedding service (text-embedding-large-exp-03-07)
    2. Fallback: If Vertex AI is unavailable, use direct Gemini API (gemini-embedding-exp-03-07)
    
    Both model variants are the same underlying model, just accessed via different APIs.
    The model supports Matryoshka embeddings, allowing for flexible output dimensionality.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        google_api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        vertex_ai_model: str = "gemini-embedding-001",  # Using a known available model by default
        gemini_api_model: str = "gemini-embedding-001",  # Accessible model from Gemini API
        dimensions: int = 1536,  # Half dimensions for this model
        output_dimensions: Optional[int] = None,  # Can request smaller dimensions if needed
        embedding_type: str = "RETRIEVAL_QUERY"  # or "retrieval_query"
    ):
        """
        Initialize the Gemini embedding reranker.
        
        Args:
            credentials_path: Path to service account credentials file for Vertex AI.
                If None, will load from VERTEXAI_SA_CREDS environment variable.
            google_api_key: API key for direct Gemini API (fallback).
                If None, will load from GOOGLE_API_KEY environment variable.
            project_id: Google Cloud project ID. If None, will be extracted from credentials.
            location: Google Cloud location (default: "us-central1")
            vertex_ai_model: Vertex AI model ID (default: "text-embedding-large-exp-03-07")
            gemini_api_model: Direct Gemini API model name (default: "gemini-embedding-exp-03-07")
            dimensions: Full embedding dimensions (default: 1536)
            output_dimensions: Optional reduced dimensions to request (Matryoshka embedding)
                If None, will use the full dimensions.
            embedding_type: Type of embedding to generate (default: "retrieval_document")
        """
        
        # Try to set up Vertex AI first
        self.vertex_ai_enabled = False
        self.gemini_api_enabled = False
        
        # Load Vertex AI credentials
        if credentials_path is None:
            credentials_path = os.getenv('VERTEXAI_SA_CREDS')
        
        # Try to set up Vertex AI prerequisites
        credentials = None
        if credentials_path and os.path.exists(credentials_path):
            try:
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                
                # Extract project ID from credentials if not provided
                if project_id is None:
                    with open(credentials_path, 'r') as f:
                        project_id = json.load(f).get('project_id')
                        if not project_id:
                            print("Warning: Couldn't extract project_id from credentials")
                
                if project_id:
                    # Just set the credentials and project ID, don't initialize here
                    # We'll initialize vertexai when we need it
                    self.vertex_ai_enabled = True
                    print(f"Initialized Vertex AI with project {project_id}")
            except Exception as e:
                print(f"Warning: Failed to initialize Vertex AI: {str(e)}")
                print(f"Will try fallback to direct Gemini API")
        
        # Set up direct Gemini API as fallback
        if google_api_key is None:
            google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if google_api_key:
            try:
                # Initialize the Gemini client with API key
                self.genai_client = genai.Client(api_key=google_api_key)
                self.gemini_api_enabled = True
                print("Initialized direct Gemini API as fallback")
            except Exception as e:
                print(f"Warning: Failed to initialize direct Gemini API: {str(e)}")
        
        # Check if at least one API is available
        if not self.vertex_ai_enabled and not self.gemini_api_enabled:
            raise ValueError(
                "Failed to initialize both Vertex AI and direct Gemini API. "
                "Please provide valid VERTEXAI_SA_CREDS or GOOGLE_API_KEY."
            )
        
        # Store configuration
        self.vertex_ai_model = vertex_ai_model
        self.gemini_api_model = gemini_api_model
        self.dimensions = dimensions
        self.output_dimensions = output_dimensions if output_dimensions else dimensions
        self.embedding_type = embedding_type
        self.location = location
        self.project_id = project_id
        self.credentials = credentials  # Store the loaded credentials
        self.credentials_path = credentials_path  # Store the path for environment variable
        
        # Initialize cache
        self.embeddings_cache = {}
        
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts using Gemini models.
        
        This will first try to use Vertex AI, and if that fails,
        it will fall back to the direct Gemini API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings
        """
        if not texts:
            return torch.zeros((0, self.output_dimensions))
        
        # Check cache for batch (to avoid redundant API calls)
        cache_key = tuple(texts)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Try Vertex AI first
        if self.vertex_ai_enabled:
            try:
                embeddings = self._get_vertex_ai_embeddings(texts, self.embedding_type)
                self.embeddings_cache[cache_key] = embeddings
                return embeddings
            except Exception as e:
                print(f"Warning: Vertex AI embedding failed: {str(e)}")
                print(f"Falling back to direct Gemini API")
                
                # Only fall back if direct API is configured
                if not self.gemini_api_enabled:
                    raise RuntimeError("Vertex AI failed and no fallback available") from e
        
        # Fall back to direct Gemini API
        if self.gemini_api_enabled:
            try:
                embeddings = self._get_direct_gemini_embeddings(texts)
                self.embeddings_cache[cache_key] = embeddings
                return embeddings
            except Exception as e:
                print(f"Error: Direct Gemini API embedding failed: {str(e)}")
                raise RuntimeError("Both Vertex AI and direct Gemini API failed") from e
        
        # This should not happen due to earlier checks, but just in case
        raise RuntimeError("No embedding provider available")
    
    def _get_vertex_ai_embeddings(self, texts: List[str], task_type: str) -> torch.Tensor:
        """
        Get embeddings using Vertex AI Embeddings API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings
        """
        try:
            # Import required modules from vertexai
            from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
            import vertexai
            
            # Initialize Vertex AI with explicit credentials
            vertexai.init(
                project=self.project_id, 
                location=self.location,
                credentials=self.credentials  # Pass the credentials explicitly
            )
            
            # Set Google application credentials for the underlying libraries
            os.environ["VERTEXAI_SA_CREDENTIALS"] = self.credentials_path
            
            # Map our embedding type to Vertex AI task type
            if task_type.lower() not in {"retrieval_document", "retrieval_query"}:
                raise ValueError(f"Gemini Embeddings task type: {task_type} is not supported. Supported types: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY")

            task_type = "RETRIEVAL_DOCUMENT"
            if self.embedding_type == "retrieval_query":
                task_type = "RETRIEVAL_QUERY"
            
            try:
                # Initialize the embedding model using from_pretrained
                model = TextEmbeddingModel.from_pretrained(self.vertex_ai_model)
                
                # Prepare input with appropriate task type
                inputs = [TextEmbeddingInput(text, task_type) for text in texts]
                
                # Prepare parameters for dimension reduction if needed
                kwargs = {}
                if self.output_dimensions != self.dimensions:
                    kwargs["output_dimensionality"] = self.output_dimensions
                
                # Get embeddings
                # gemini-embedding-001 only supports single input text per request
                if "gemini" in self.vertex_ai_model.lower():
                    embeddings_response = []
                    for input_text in inputs:
                        response = model.get_embeddings([input_text], **kwargs)
                        embeddings_response.extend(response)
                else:
                    # Other models support batch processing up to 250 texts
                    embeddings_response = model.get_embeddings(inputs, **kwargs)
            except Exception as e:
                # If the model name isn't recognized, try a fallback model
                if "No model was found" in str(e) or "not found" in str(e):
                    print(f"Model {self.vertex_ai_model} not found, trying textembedding-gecko@latest")
                    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
                    
                    # Get embeddings with the fallback model
                    # textembedding-gecko supports batch processing
                    embeddings_response = model.get_embeddings(inputs, **kwargs)
                else:
                    # Re-raise if it's not a model not found error
                    raise
            
            # Extract and convert to tensor
            embeddings = []
            for response in embeddings_response:
                embeddings.append(response.values)
            
            return torch.tensor(embeddings)
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import Vertex AI modules: {str(e)}. Make sure vertexai is installed.") from e
    
    def _get_direct_gemini_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings using direct Gemini API with the google-genai package.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings
        """
        # Map embedding types to Gemini task types
        task_type = "RETRIEVAL_DOCUMENT"
        if self.embedding_type.lower() == "retrieval_query":
            task_type = "RETRIEVAL_QUERY"
        
        # Configure embedding request
        config = EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.output_dimensions
        )
        
        # Process texts in batches to avoid API limits
        BATCH_SIZE = 10  # Adjust as needed
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            
            # Call the Gemini API to get embeddings
            response = self.genai_client.models.embed_content(
                model=self.gemini_api_model,
                contents=batch,
                config=config
            )
            
            # Extract embeddings from response
            for embedding_obj in response.embeddings:
                all_embeddings.append(embedding_obj.values)
        
        return torch.tensor(all_embeddings)
    
    def calculate_scores(
        self,
        queries: List[str],
        documents: List[str],
        normalize: str = "softmax"
    ) -> torch.Tensor:
        """
        Calculate similarity scores between queries and documents.
        
        This overrides the base class method to handle different embedding types
        for queries and documents.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            normalize: Normalization method
            
        Returns:
            torch.Tensor of shape (num_queries, num_documents) containing similarity scores
        """
        # Save original embedding type
        original_type = self.embedding_type
        
        try:
            # Set to query type for queries
            self.embedding_type = "retrieval_query"
            query_embeddings = self._get_embeddings(queries)
            
            # Set to document type for documents
            self.embedding_type = "retrieval_document"
            doc_embeddings = self._get_embeddings(documents)
            
            # Calculate similarity scores
            scores = query_embeddings @ doc_embeddings.T
            
            # Apply normalization
            if normalize == "softmax":
                scores = torch.softmax(scores, dim=-1)
            elif normalize == "scale":
                scores = scores * 100
                
            return scores
            
        finally:
            # Restore original embedding type
            self.embedding_type = original_type