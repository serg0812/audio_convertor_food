from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
from langchain.chains.openai_functions import (
    create_openai_fn_chain, create_structured_output_chain)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools import StructuredTool

class FoodDetails(BaseModel):
    """
    Pydantic arguments schema for food details
    """
    food: str = Field(..., description="item from the menu person would like to order e.g. chicken wings")
    qty: str = Field(..., description="how many of items person would like to order")

class DrinkDetails(BaseModel):
    """
    Pydantic arguments for drink details
    """
    drink: str = Field(..., description="what drink person would like to order e.g. beer, wine")
    qty: str = Field(..., description="how many of items person would like to order")

def get_food_details(food: str, qty: str) -> str:
    response = FoodDetails(food=food, qty=qty)
    
    return response.json()

def get_drink_details(drink: str, qty: str) -> str:
    response = DrinkDetails(drink=drink, qty=qty)
    
    return response.json()


# Define a main function to process text from Streamlit
def process_text_from_streamlit(text_output: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4-0125-preview",
        response_format={"type": "json_object"}
    )

    tools = [
        StructuredTool.from_function(func=get_food_details, args_schema=FoodDetails, description="Function to get food details"),
        StructuredTool.from_function(func=get_drink_details, args_schema=DrinkDetails, description="Function to get drink details")
    ]
    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    system_init_prompt = "You are accepting the order in ther restaurant"
    user_init_prompt = f"Find all details about the food and drink person would like to order and return the output in json in french, this is the detailed text description: {text_output}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_init_prompt),
        ("user", user_init_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = (
        {"input": lambda x: x["input"],
         "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": text_output})
    return response.get("output")
