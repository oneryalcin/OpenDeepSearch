
import os
from smolagents import CodeAgent, GradioUI, LiteLLMModel
from opendeepsearch import OpenDeepSearchTool

# os.environ['VERTEXAI_SA_CREDS'] = ''

#
# from google import genai
#
# client = genai.Client(api_key="GEMINI_API_KEY")
#
# result = client.models.embed_content(
#         model="gemini-embedding-exp-03-07",
#         contents="How does alphafold work?",
# )


# SERPER_API_KEY, GEMINI_API_KEY, JINA_API_KEY must be set in the environment before running this script
# # Or for SearXNG
# # os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# # os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"


search_tool = OpenDeepSearchTool(
    model_name='gemini/gemini-2.5-flash-lite',
    # model_name='openai/gpt-4.1-mini',
    reranker='gemini',
    search_provider='serper',
    serper_api_key=os.getenv("SERPER_API_KEY")
)


# response = search_tool.forward("how long would a cheetah at full speed take to run the length of Pont Alexandre III?")

code_agent = CodeAgent(
    tools=[search_tool],
    model=LiteLLMModel(model_id="gemini/gemini-2.0-flash"),
    # model=LiteLLMModel(model_id="openai/gpt-4.1-mini"),
    additional_authorized_imports=["numpy", "pandas", "matplotlib"],
    max_steps=25,
)

result = code_agent.run("How long would a cheetah at full speed take to run the length of Pont Alexandre III?")
print(result)

"""
Example run logs:
╭────────────────────────────────── New run ───────────────────────────────────╮
│                                                                              │
│ How long would a cheetah at full speed take to run the length of Pont        │
│ Alexandre III?                                                               │
│                                                                              │
╰─ LiteLLMModel - openai/gpt-4.1-mini ─────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ───────────────────────────────────────────────────── 
  pont_length_info = web_search(query="length of Pont Alexandre III")           
  print(pont_length_info)                                                       
 ────────────────────────────────────────────────────────────────────────────── 
Using Jina Reranker
Execution logs:
The length of Pont Alexandre III is approximately 160 meters (520 feet).
Out: None
[Step 1: Duration 7.78 seconds| Input tokens: 2,015 | Output tokens: 133]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ───────────────────────────────────────────────────── 
  cheetah_speed_info = web_search(query="top speed of a cheetah in meters per   
  second")                                                                      
  print(cheetah_speed_info)                                                     
 ────────────────────────────────────────────────────────────────────────────── 
Using Jina Reranker
Execution logs:
The top speed of a cheetah is reliably measured at 29 meters per second.
Out: None
[Step 2: Duration 5.91 seconds| Input tokens: 4,255 | Output tokens: 214]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ───────────────────────────────────────────────────── 
  pont_length_meters = 160                                                      
  cheetah_speed_mps = 29                                                        
                                                                                
  time_seconds = pont_length_meters / cheetah_speed_mps                         
  final_answer(time_seconds)                                                    
 ────────────────────────────────────────────────────────────────────────────── 
Out - Final answer: 5.517241379310345
[Step 3: Duration 2.76 seconds| Input tokens: 6,681 | Output tokens: 323]
"""