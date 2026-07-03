from typing import Optional, Literal
from smolagents import Tool
from opendeepsearch.ods_agent import OpenDeepSearchAgent

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: Literal["infinity", "jina", "vertex", "gemini"] = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
        # Additional parameters for Vertex AI and Gemini Embedding rerankers
        vertex_credentials_path: Optional[str] = None,
        vertex_project_id: Optional[str] = None,
        google_api_key: Optional[str] = None,
        output_dimensions: Optional[int] = None  # For Gemini embeddings
    ):
        super().__init__()
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key
        
        # Store additional parameters for Google rerankers
        self.vertex_credentials_path = vertex_credentials_path
        self.vertex_project_id = vertex_project_id
        self.google_api_key = google_api_key
        self.output_dimensions = output_dimensions
        
        # For tracking initialization status
        self.is_initialized = False

    def forward(self, query: str):
        if not self.is_initialized:
            raise ValueError("Tool not initialized. Call setup() before using forward().")
        answer = self.search_tool.ask_sync(query, max_sources=2, pro_mode=True)
        return answer

    def setup(self):
        """Initialize the search tool with the configured parameters."""
        # Create parameters dictionary with common settings
        params = {
            "model": self.search_model_name,
            "reranker": self.reranker,
            "search_provider": self.search_provider,
            "serper_api_key": self.serper_api_key,
            "searxng_instance_url": self.searxng_instance_url,
            "searxng_api_key": self.searxng_api_key
        }
        
        # Add Google-specific parameters for Vertex and Gemini rerankers
        source_processor_config = {}
        if self.reranker in ["vertex", "gemini"]:
            source_processor_config["reranker"] = self.reranker
            
            if self.vertex_credentials_path:
                source_processor_config["credentials_path"] = self.vertex_credentials_path
            
            if self.vertex_project_id:
                source_processor_config["project_id"] = self.vertex_project_id
                
            if self.google_api_key:
                source_processor_config["google_api_key"] = self.google_api_key
                
            if self.output_dimensions and self.reranker == "gemini":
                source_processor_config["output_dimensions"] = self.output_dimensions
            
            params["source_processor_config"] = source_processor_config
            
        # Create the search tool
        self.search_tool = OpenDeepSearchAgent(**params)
        self.is_initialized = True