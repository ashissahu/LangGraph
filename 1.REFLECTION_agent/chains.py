#uv init
#uv venv

# uv add langchain langchain-openai langgraph python-dotenv black isort

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# reflection_prompt = ChatPromptTemplate.from_messages([...])
#Creates a chat prompt template for the reflection task.
#MessagesPlaceholder(variable_name="messages") means the prompt can later accept user or assistant messages dynamically.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#generation_prompt = ChatPromptTemplate.from_messages([...])
#Creates a second chat prompt template for tweet generation.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI() #Initislizing the LLM for chain 
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# What this code is set up to do
# generate_chain is for producing a tweet from a user request.
# reflect_chain is for grading or critiquing a tweet and giving improvement recommendations.
# Both chains use the same underlying ChatOpenAI model, but with different system instructions.