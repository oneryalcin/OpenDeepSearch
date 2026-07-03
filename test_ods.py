import os
import asyncio
from src.opendeepsearch.ods_agent import OpenDeepSearchAgent

# SERPER_API_KEY, JINA_API_KEY must be set in the environment before running this script
# # Or for SearXNG
# # os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# # os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"

# async def test():
#     print("Creating agent...")
#     agent = OpenDeepSearchAgent(model="openrouter/google/gemini-2.0-flash-001", reranker="jina")
#     print("Agent created, searching for information...")
#     context = await agent.search_and_build_context('What are the earnings of Nvidia in 2024 Q1', max_sources=4)
#     print("Successfully built context:")
#     print(context[:500])  # Print first 500 chars of context

# if __name__ == "__main__":
#     asyncio.run(test()) 


from src.opendeepsearch.ods_agent import OpenDeepSearchAgent
from smolagents import CodeAgent

async def run_agent():
    # Initialize OpenDeepSearchAgent
    search_agent = OpenDeepSearchAgent(
        model="openrouter/google/gemini-2.0-flash-001",
        reranker="jina"
    )
    
    # Create CodeAgent with search capabilities
    code_agent = CodeAgent(
        tools=[search_agent.search_and_build_context],
        model="openrouter/google/gemini-2.0-flash-001"
    )
    
    # Run coding task with search capabilities
    result = await code_agent.run(
        "Can you please analyze Nvidia's Q1 2024 earnings"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(run_agent()) 
