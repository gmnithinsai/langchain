# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:47:59 2024

@author: gmnit
"""

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
import datetime
import requests
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


# os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(temperature=0,model = "gpt-4o-mini")
streamlit_memory = StreamlitChatMessageHistory(key="chat_messages")
if len(streamlit_memory.messages) == 0:
    streamlit_memory.add_ai_message("How can I help you?")



# Define your desired data structure
class Journey(BaseModel):
    journey_date: str = Field(description="date of the journey and date format should be {}-mm-dd".format(datetime.datetime.now().year))
    destination_name: str = Field(description="destination of the journey")
    source_name: str = Field(description = "source or origin of the journey")
    
class Seat(BaseModel):
    train_number: str = Field(description="number of the train")

  

def trainBetweenStations(source,destination,journey_date):
    url = "https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"
    querystring = {"fromStationCode":source,"toStationCode":destination,"dateOfJourney":journey_date}
    headers = {
    	"x-rapidapi-key": "72a49711f4msh148bbea9230845ap1162bcjsn6929cea11bac",
    	"x-rapidapi-host": "irctc1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()


def format_railyway_json_response(railway_api_response):
    prompt_template = ChatPromptTemplate(messages = [
        ("system", "You are a railway AI assistant and your task is to answer the customer query about the Indian railways from the information given to you. Information: {railway_info}."),
        ("user", " {query}")
    ],input_variables = ['railway_info'])
    chain = prompt_template | model 
    response = chain.invoke({"query": st.session_state.query,"railway_info":railway_api_response})
    return response

def check_train_between_stations_tool_flow(query_params):
    parser = PydanticOutputParser(pydantic_object=Journey)
    prompt = PromptTemplate(
        template="Given the query extract the fields such as journey_date,destination_name,source_name.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | model.with_structured_output(Journey)
    query_params = chain.invoke({"query": st.session_state.query})
    
    station_codes ={
      "Mumbai": "CST",
      "Delhi": "NDLS",
      "Kolkata": "HWH",
      "Chennai": "MAS",
      "Bangalore": "SBC",
      "Hyderabad": "HYB",
      "Ahmedabad": "ADI",
      "Pune": "PUNE",
      "Jaipur": "JP",
      "Lucknow": "LKO",
      "Patna": "PNBE",
      "Bhopal": "BPL",
      "Amritsar": "ASR",
      "Nagpur": "NGP",
      "Thiruvananthapuram": "TVC",
      "Mysore":"MYS"
    }
    st.session_state['journey_date'] = query_params.journey_date
    st.session_state['source_name'] = query_params.source_name
    st.session_state['destination_name'] = query_params.destination_name
    
    source_station_code = station_codes.get(query_params.source_name)
    destination_station_code = station_codes.get(query_params.destination_name)

    trains_between_stations_json_response = trainBetweenStations(source_station_code,destination_station_code,query_params.journey_date)
    return trains_between_stations_json_response


def checkSeatAvailability(source,destination,train_number,date):
    url = "https://irctc1.p.rapidapi.com/api/v1/checkSeatAvailability"
    quota = 'GN'
    querystring = {"fromStationCode":source,"quota":quota,"toStationCode":destination,"trainNo":train_number,"date":date,"classType" :'SL'}
    headers = {
     	"x-rapidapi-key": "72a49711f4msh148bbea9230845ap1162bcjsn6929cea11bac",
     	"x-rapidapi-host": "irctc1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def check_seat_availability_tool_flow(query_params):
    # Set up a parser + inject instructions into the prompt template
    parser = PydanticOutputParser(pydantic_object=Seat)
    prompt = PromptTemplate(
        template="Given the query extract the field train_number.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | model.with_structured_output(Seat)
    query_params = chain.invoke({"query": st.session_state.query})
    response = checkSeatAvailability(st.session_state['source_name'], st.session_state.destination_name, query_params.train_number,st.session_state['journey_date'])
    return response





def rail_agent():
    tools = [
        Tool.from_function(
            name="Check Seat Availability",
            description="Use when you need to check the availability of seats. The question will contain the train number. Returns a json.",
            func=check_seat_availability_tool_flow,
        ),
        Tool.from_function(
            name="Search train between stations",
            description="Use when the query is related to search train between the stations.The question will include a source and destination cities and journey date.Return output in string",
            func=check_train_between_stations_tool_flow,
        ),
    ]
    
    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)  
    response = agent_executor.invoke({"input":st.session_state.query,"chat_history":streamlit_memory.messages})
    return response
    

def streamlit_ui():
    st.title("RAIL Assistant")
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4.o-mini"
    if "query" not in st.session_state:
        st.session_state['query'] = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in streamlit_memory.messages:
        st.chat_message(message.type).write(message.content)
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.query = prompt
        streamlit_memory.add_user_message(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Typing..."):
                result = rail_agent()
                st.markdown(result['output'])
                
            # st.session_state.messages.append({"role": "assistant", "content": result['output']})
            streamlit_memory.add_ai_message(result['output'])
            
            
streamlit_ui()
 
# import requests

# def searchStation(station_code):
#     url = "https://irctc1.p.rapidapi.com/api/v1/searchStation"
#     querystring = {"query":station_code}
#     headers = {
#     	"x-rapidapi-key": "6281fc2406msh88e05da2c9503ccp11167ajsncd56d6fb7761",
#     	"x-rapidapi-host": "irctc1.p.rapidapi.com"
#     }
#     response = requests.get(url, headers=headers, params=querystring)
#     return response.json()
    
# station_code = result.source_name
# result = searchStation(station_code)

# def searchTrain(train_number):
#     url = "https://irctc1.p.rapidapi.com/api/v1/searchTrain"
#     querystring = {"query":train_number}
#     headers = {
#     	"x-rapidapi-key": "6281fc2406msh88e05da2c9503ccp11167ajsncd56d6fb7761",
#     	"x-rapidapi-host": "irctc1.p.rapidapi.com"
#     }
#     response = requests.get(url, headers=headers, params=querystring)
#     return response.json()


# def getPNRStatus(pnr_number):
#     url = "https://irctc1.p.rapidapi.com/api/v3/getPNRStatus"
#     querystring = {"pnrNumber":pnr_number}
#     headers = {
#     	"x-rapidapi-key": "6281fc2406msh88e05da2c9503ccp11167ajsncd56d6fb7761",
#     	"x-rapidapi-host": "irctc1.p.rapidapi.com"
#     }
#     response = requests.get(url, headers=headers, params = querystring )
#     return response.json()

#     return response.json()
# """