from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [TavilySearch()]

agent = create_agent(
    model=llm, 
    tools=tools, 
    response_format=AgentResponse
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 job postings for an AI Engineer using Langchain in Rio de Janeiro on LinkedIn and list their details."
                }
            ]
        }
    )
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
