import torch
from typing import List, Optional
from pylate_rs import models
from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher

class PyLateReranker(BaseSemanticSearcher):
    """
    A semantic reranker using PyLate-rs for high-performance ColBERT inference.
    
    This reranker uses late interaction (token-level) embeddings for more accurate
    relevance scoring compared to traditional bi-encoder approaches.
    
    Attributes:
        model_name (str): Name or path of the ColBERT model to use
        device (str): Device to run the model on ('cpu', 'cuda', 'mps')
        
    Example:
        ```python
        reranker = PyLateReranker(
            model_name="answerdotai/answerai-colbert-small-v1",
            device="cuda"
        )
        
        documents = ["Paris is the capital of France.", "London is the capital of UK."]
        reranked_docs = reranker.get_reranked_documents(
            query="What is the capital of France?",
            documents=documents,
            top_k=1
        )
        ```
    """
    
    def __init__(
        self, 
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        device: Optional[str] = None
    ):
        """
        Initialize the PyLate reranker with a ColBERT model.
        
        Args:
            model_name: HuggingFace model ID or local path to a ColBERT model
            device: Device to run on. If None, will auto-detect (cuda > mps > cpu)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        
        # Initialize the ColBERT model
        print(f"Loading PyLate model: {model_name} on {device}")
        self.model = models.ColBERT(
            model_name_or_path=model_name,
            device=device
        )
        
    def _get_embeddings(self, texts: List[str], is_query: bool = True):
        """
        Get ColBERT embeddings for texts.
        
        Args:
            texts: List of text strings to embed
            is_query: Whether these are query texts (True) or document texts (False)
            
        Returns:
            Embeddings (as returned by pylate-rs, typically numpy arrays)
        """
        # PyLate-rs returns embeddings as numpy arrays
        embeddings = self.model.encode(
            sentences=texts,
            is_query=is_query
        )
        
        return embeddings
    
    def calculate_scores(
        self,
        queries: List[str],
        documents: List[str],
        normalize: str = "softmax"
    ) -> torch.Tensor:
        """
        Calculate similarity scores between queries and documents using ColBERT.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            normalize: Normalization method ('softmax', 'scale', or None)
            
        Returns:
            torch.Tensor of shape (num_queries, num_documents) containing similarity scores
        """
        # Encode queries and documents
        query_embeddings = self._get_embeddings(queries, is_query=True)
        doc_embeddings = self._get_embeddings(documents, is_query=False)
        
        # Calculate similarity scores using ColBERT's MaxSim
        # PyLate-rs provides a similarity method for this
        scores = self.model.similarity(query_embeddings, doc_embeddings)
        
        # Convert to torch tensor
        scores = torch.tensor(scores)
        
        # Apply normalization
        if normalize == "softmax":
            scores = torch.softmax(scores, dim=-1)
        elif normalize == "scale":
            # Scale scores to a more interpretable range
            scores = scores * 100
            
        return scores
    
    def get_reranked_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> str:
        """
        Rerank documents based on their relevance to the query.
        
        This method overrides the base class to provide more efficient reranking
        for single queries (the most common use case).
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            String containing the top-k most relevant documents
        """
        if not documents:
            return ""
            
        # Get scores for the single query
        scores = self.calculate_scores([query], documents, normalize=None)
        
        # Get the scores for our single query (first row)
        query_scores = scores[0]
        
        # Get top-k indices
        top_k = min(top_k, len(documents))
        top_indices = torch.topk(query_scores, top_k).indices.tolist()
        
        # Return the top documents
        top_documents = [documents[i] for i in top_indices]
        return "\n\n".join(top_documents)