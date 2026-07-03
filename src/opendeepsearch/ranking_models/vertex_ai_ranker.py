import os
import json
import torch
import importlib.util
from typing import List, Optional, Dict, Union
from dotenv import load_dotenv

# Import required for authentication
from google.oauth2 import service_account

# We'll import these dynamically to handle import errors gracefully
# from google.cloud import discoveryengine_v1 as discoveryengine
# from google.api_core.client_options import ClientOptions

from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher

class VertexAIRanker(BaseSemanticSearcher):
    """
    Cross-encoder semantic reranker using Google Cloud Vertex AI ranking service.
    
    This reranker directly evaluates query-document pairs for relevance,
    unlike bi-encoder approaches that compute separate embeddings.
    """
    
    def __init__(
        self, 
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "global",
        model: str = "semantic-ranker-default-004"
    ):
        """
        Initialize the Vertex AI reranker.
        
        Args:
            credentials_path: Path to service account credentials file.
                If None, will load from VERTEXAI_SA_CREDS environment variable.
            project_id: Google Cloud project ID. If None, will be extracted from credentials.
            location: Google Cloud location (default: "global")
            model: Model name to use (default: "semantic-ranker-default-004")
        """
        # Load credentials
        if credentials_path is None:
            load_dotenv()
            credentials_path = os.getenv('VERTEXAI_SA_CREDS')
            if not credentials_path:
                raise ValueError("No credentials path provided and VERTEXAI_SA_CREDS not found in environment variables")
        
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Extract project ID from credentials if not provided
        if project_id is None:
            with open(credentials_path, 'r') as f:
                project_id = json.load(f).get('project_id')
                if not project_id:
                    raise ValueError("No project_id provided and couldn't extract it from credentials")
        
        # Store configuration for delayed initialization
        self.project_id = project_id
        self.location = location
        self.credentials = credentials
        self.credentials_path = credentials_path  # Store the path for environment variable
        self.is_initialized = False
        
        # We'll initialize the client when needed in _ensure_client()
        
        self.model = model
        
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        This method is required by the BaseSemanticSearcher interface but is not used.
        
        The VertexAIRanker is a cross-encoder that evaluates (query, document) pairs directly,
        rather than computing separate embeddings.
        
        Args:
            texts: List of text strings (not used)
            
        Returns:
            Empty tensor (not used in actual ranking)
        """
        # This is just a placeholder to satisfy the interface
        # In practice, we override rerank() to use the Vertex AI ranking API directly
        return torch.zeros((len(texts), 1))
    
    def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5,
        normalize: str = "softmax"
    ) -> Union[List[Dict[str, Union[str, float]]], List[List[Dict[str, Union[str, float]]]]]:
        """
        Rerank documents based on their relevance to the query using Vertex AI.
        
        This method overrides the base class method to use the Vertex AI ranking API directly.
        
        Args:
            query: Query string or list of query strings
            documents: List of documents to rerank
            top_k: Number of top results to return per query
            normalize: Ignored for this implementation
            
        Returns:
            List of dicts containing reranked documents and their scores.
        """
        if isinstance(query, list):
            # Handle multi-query case
            results = []
            for q in query:
                results.append(self._rank_single_query(q, documents, top_k))
            return results
        else:
            # Handle single query case
            return self._rank_single_query(query, documents, top_k)
    
    def _ensure_client(self):
        """Initializes the client if not already initialized."""
        if not self.is_initialized:
            try:
                # Import required modules
                from google.cloud import discoveryengine_v1 as discoveryengine
                from google.api_core.client_options import ClientOptions
                
                # Set Google application credentials for the underlying libraries
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                
                # Initialize client
                self.parent = f"projects/{self.project_id}/locations/{self.location}"
                client_options = ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com")
                self.client = discoveryengine.RankingServiceClient(
                    credentials=self.credentials,
                    client_options=client_options
                )
                
                # Store modules for later use
                self.discoveryengine = discoveryengine
                self.is_initialized = True
                
            except ImportError as e:
                raise RuntimeError(f"Failed to import Google Cloud Discovery Engine: {e}. Make sure the google-cloud-discoveryengine package is installed.") from e
    
    def _rank_single_query(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Rank documents for a single query using Vertex AI.
        
        Args:
            query: Query string
            documents: List of documents to rank
            top_k: Number of top results to return
            
        Returns:
            List of dicts with documents and scores
        """
        # Make sure client is initialized
        self._ensure_client()
        
        # Create the request
        request = self.discoveryengine.RankRequest(
            parent=self.parent,
            query=query,
            model_id=self.model,
        )
        
        # Add documents to the request
        for i, doc in enumerate(documents):
            request.documents.append(
                self.discoveryengine.RankingDocument(
                    id=str(i),
                    content=self.discoveryengine.RankingContent(
                        document=doc
                    )
                )
            )
        
        try:
            # Call the API
            response = self.client.rank(request)
            
            # Process and return results
            results = []
            for result in response.results[:min(top_k, len(documents))]:
                doc_id = int(result.id)
                results.append({
                    "document": documents[doc_id],
                    "score": result.ranking_score
                })
            
            return results
            
        except Exception as e:
            # Handle errors
            print(f"Error ranking with Vertex AI: {str(e)}")
            # Fall back to original order with neutral scores
            return [{"document": doc, "score": 0.5} for doc in documents[:top_k]]
    
    def get_reranked_documents(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5,
        normalize: str = "softmax"
    ) -> Union[str, List[str]]:
        """
        Returns only the reranked documents without scores.
        
        Args:
            query: Query string or list of query strings
            documents: List of documents to rerank
            top_k: Number of top results to return per query
            normalize: Ignored for this implementation
            
        Returns:
            For single query: Joined string of reranked document strings
            For multiple queries: List of joined strings of reranked document strings
        """
        results = self.rerank(query, documents, top_k, normalize)
        
        if isinstance(query, list):
            # Handle multi-query case
            return ["\n".join([item["document"].strip() for item in result]) for result in results]
        else:
            # Handle single query case
            return "\n".join([item["document"].strip() for item in results])