import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun


api_key = os.getenv("GROQ_API_KEY")
groq_api_key = api_key.strip() if api_key else None;
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not configured")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str):
    """Search cricket match data from ESPN and Cricbuzz."""
    return search.run(query)

agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="""
You are a cricket AI analyst.

Give:
- tactical reasoning
- match insights
- final YES/NO
"""
)
