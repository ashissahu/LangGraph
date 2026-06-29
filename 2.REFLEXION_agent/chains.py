#Langsmith Tracing : https://smith.langchain.com/public/6d38eef7-f5f0-4a27-bbfe-5a2fb2e56afb/r/019f11c4-cc51-72f3-84f7-9b58672db9ad

#Step 1: Setup & Imports
from dotenv import load_dotenv

load_dotenv()
import datetime

from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from cool_classes import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="gpt-4")
#JSON Parser → converts response to dictionary
#Pydantic Parser → converts to structured object
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

#Step 2: Define the Goal of the Agent
#The Actor Agent (First Responder) should:
#Take a user query/topic
#Generate:
#✍️ A 250-word article (first draft)
#🔍 A critique (feedback)
#🔎 Search queries to improve it

#Build a ChatPromptTemplate with:System Prompt,Message History Placeholder
# Prompt should include:Role: Expert researcher,Dynamic time injection,3 instructions:Write a 250-word essay,Critique your answer (strict/improving),Suggest search queries

#Add Message History Support -Use MessagesPlaceholder-This allows:Iterative improvements,Reuse in future (Revisor Agent)

#Inject Dynamic Time -Use .partial():Add current date/time dynamically,Use a lambda function,Format: ISO format
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

 #Bind tool:AnswerQuestion schema, Force tool usage:tool_choice = AnswerQuestion
first_responder = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
) | llm.bind_tools(tools=[AnswerQuestion],
                   tool_choice="AnswerQuestion")



#Create Revision Instructions
#Define a new instruction template:
#Include these rules:
#1.Revise the previous answer using new information
#2.Use critique to:
#Add missing information
#Remove unnecessary (superfluous) content
#3.Keep answer within 250 words
#4.Add numerical citations (e.g., [1], [2])
#5.Add a References section at the end (URLs)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

#Reuse the existing Actor Prompt Template
#Replace:  First Instruction Placeholder 

#Create a new chain (Reviser Chain):
#Use: Actor Prompt Template
#Inject: Revision Instructions
#Pass into: GPT-4

#The Reviser Agent will: 1.Read previous draft ,2.Analyze critique:Add missing info,Remove fluff
#3.Use search results:Add real-world data 4.Generate:Updated answer,Citations ([1], [2]),Reference links
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")



if __name__ == "__main__":
    human_message = HumanMessage(
        content = "Write about AI Powered SOC/autonomus soc problem domain,"
        "list startups that do that and raise capital."
    )

    chain = (
        actor_prompt_template.partial(
            first_instruction="Provide a detailed ~250 word answer."
            ) | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")
            | parser_pydantic
    )

    res = chain.invoke(input={"messages":[human_message]})
    print(res)