from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# 1) TOOLS
# ==================================================

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> dict:
    """Get current weather for a city"""
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"
    return requests.get(url).json()


tools = [search_tool, get_weather_data]


# ==================================================
# 2) GROQ MODEL WITH TOOL CALLING
# ==================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Attach tools to the model
llm_with_tools = llm.bind_tools(tools)


# ==================================================
# 3) SIMPLE AGENT LOOP
# ==================================================

def run_agent(question: str):

    messages = [
        HumanMessage(content=question)
    ]

    # First LLM call
    response = llm_with_tools.invoke(messages)

    # If model wants to use a tool
    if isinstance(response, AIMessage) and response.tool_calls:
        for call in response.tool_calls:

            tool_name = call["name"]
            args = call["args"]

            # Execute tool
            for t in tools:
                if t.name == tool_name:
                    tool_result = t.invoke(args)

                    messages.append(response)
                    messages.append(
                        HumanMessage(
                            content=f"Tool result: {tool_result}"
                        )
                    )

                    # Final answer with tool result
                    final = llm.invoke(messages)
                    return final.content

    # If no tool needed
    return response.content


# ==================================================
# 4) RUN
# ==================================================

if __name__ == "__main__":

    q = "What is the current temp of pune"

    ans = run_agent(q)

    print("\nFINAL ANSWER:\n")
    print(ans)
