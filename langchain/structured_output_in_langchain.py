# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:12:55 2024

@author: gmnit
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(temperature=0,model = "gpt-4o-mini")


# Example 1: Using String PromptTemplates
prompt_template = PromptTemplate.from_template(
    "You are a customer support assistant and your task is to provide the sentiment and summarization of the user remarks."
    "Return summary and sentiment in JSON format. Below is the remarks provided by customer:{remarks}"
)
chain = prompt_template | llm
response = chain.invoke({
    "remarks": "Hi, I have ordered a peter england white t-shirt, a brand new one but I received a brown t-shirt with some stinks on it and the packaging is also not proper. I just wanted a complete refund for my order and I am really disappointed about the service you gave."
})
print(response.content)  # The response is a string in JSON format

# Example 2: Using JsonOutputParser
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support assistant and your task is to provide the sentiment and summarization of the user remarks. Return summary and sentiment in JSON format."),
    ("user", "{query}")
])

chain = prompt_template | llm | JsonOutputParser()
response = chain.invoke({
    "query": "Hi, I have ordered a peter england white t-shirt, a brand new one but I received a brown t-shirt with some stinks on it and the packaging is also not proper. I just wanted a complete refund for my order and I am really disappointed about the service you gave."
})
print(response)  # The response is a JSON object

# Example 3: Using PydanticOutputParser for structured output
class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

structured_llm = llm.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about cats")

# Define your desired data structure
class TrainQuery(BaseModel):
    journey_date: str = Field(description="Date of the journey, in yyyy-mm-dd format")
    destination_name: str = Field(description="Destination of the journey")
    source_name: str = Field(description="Source or origin of the journey")

    # Custom validation for destination_name
    @validator("destination_name")
    def validate_destination(cls, field):
        if field.lower() == 'unknown':
            raise ValueError("Please, mention the destination!")
        return field

# Query intended to prompt the language model
joke_query = "Give me trains from Mumbai to Chennai for Aug 12."

# Set up a parser + inject instructions into the prompt template
parser = PydanticOutputParser(pydantic_object=TrainQuery)
prompt = PromptTemplate(
    template="Given the query, extract the fields such as journey_date, destination_name, source_name.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(temperature=0)
chain = prompt | model.with_structured_output(TrainQuery)
result = chain.invoke({"query": joke_query})

print(result)
