from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=react_prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        {
            "input": {
                "Search for 3 job postings for an AI Engineer using Langchain in Rio de Janeiro on LinkedIn and list their details."
            }
        }
    )
    print(result)


if __name__ == "__main__":
    main()
