# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:36:47 2024

@author: gmnit
"""

from typing import Annotated
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

os.environ['ANTHROPIC_API_KEY'] = ""
os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""


memory = SqliteSaver.from_conn_string(":memory:")

llm = ChatAnthropic(model="claude-3-haiku-20240307")

search = GoogleSearchAPIWrapper()
tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)


while True:
    config = {"configurable": {"thread_id": "1"}}

    user_input = input("Query:")
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
   
            
   
"""

OUTPUT:
    
    
Query:my name is nithin
================================ Human Message =================================

my name is nithin
================================== Ai Message ==================================

It's nice to meet you, Nithin! I'm an AI assistant here to help. Since you introduced yourself, is there anything I can assist you with today?
Query:Do you know me
================================ Human Message =================================

Do you know me
================================== Ai Message ==================================

No, I don't actually know you personally. As an AI assistant, I don't have any prior knowledge about specific individuals. I'm happy to get to know you better through our conversation, but I don't have any pre-existing information about you. Please feel free to tell me more about yourself if you'd like!
Query:Do you know my name
================================ Human Message =================================

Do you know my name
================================== Ai Message ==================================

No, I don't know your name beyond what you've already told me - that your name is Nithin. As an AI system, I don't have any pre-existing knowledge about you or other specific individuals. I only know what is provided to me during our conversation. Could you tell me more about yourself if you'd like me to get to know you better?
"""

   