from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text of a text by characters"""
    print(f"get_text_length entered with text: {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # Stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain (.bind_tools)!")
    tools = [get_text_length]

    llm = ChatOpenAI(model="gpt-4o", temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm.bind_tools(tools)

    # Start the conversation
    messages = [HumanMessage(content="What is the length of the text 'Strawberry'?")]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return the results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"{observation=}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        print(ai_message.content)
        break
