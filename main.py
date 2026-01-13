from typing import Union, List
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool, tool
from langchain_core.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_openai import ChatOpenAI

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text of a text by characters"""
    print(f"get_text_length entered with text: {text=}")
    text = text.strip("'\n").strip('"') # Stripping away non alphabetic characters just in case
    
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]
    
    prompt = PromptTemplate.from_template(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS).partial(
            tools=render_text_description(tools),
            tool_names=", ".join([tool.name for tool in tools]
        )
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, stop=["\nObservation", "Observation:"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []
    
    agent = prompt | llm | ReActSingleInputOutputParser()
    
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({
            "input": "What is the length of the text 'Strawberry'?",
            "agent_scratchpad": format_log_to_str(intermediate_steps)
        })
        print(agent_step)
        
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            tool_to_use = find_tool_by_name(tools, tool_name)
            
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))
    
    if isinstance(agent_step, AgentFinish):
        print(f"{agent_step.return_values=}")