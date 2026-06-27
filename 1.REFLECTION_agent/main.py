#LANGSMITH TRACING :https://smith.langchain.com/public/a6358cb2-59ed-4241-9124-b2991cd15f27/r/7dc418ec-0ee5-4767-90bb-a69a8ab7aa58
"""
LangGraph Generate-Reflect Loop Example
---------------------------------------

This script builds a self-improving AI workflow using LangGraph.

Flow:
    1. User provides input
    2. GENERATE node improves content
    3. REFLECT node critiques it
    4. Loop continues until stopping condition
    5. Final improved output is returned

This demonstrates a basic "AI Agent" pattern.
"""

# ==============================
# 1. IMPORTS
# ==============================

from typing import TypedDict, Annotated  # For defining structured state
#TypedDict → Defines structured state (like schema)
#Annotated → Adds metadata (used by LangGraph)

from dotenv import load_dotenv  # Loads environment variables (API keys etc.)
load_dotenv()

# Message abstractions used by LangChain / LangGraph
from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage → Generic message type
# HumanMessage → Represents user input

# Core LangGraph components
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
# StateGraph → Core LangGraph builder
# END → Special node to terminate graph
# add_messages → Handles message accumulation in state ( The entire goal of this function is to ensure new messages are appended to the existing conversation history instead of replacing it)

# Custom chains (LLM pipelines)
# generate_chain → creates/improves output
# reflect_chain → critiques output
from chains import generate_chain, reflect_chain


# ==============================
# 2. STATE DEFINITION - This defines the state schema that flows through your graph
# ==============================

class MessageGraph(TypedDict):
    """
    Defines the structure of the state that flows through the graph.

    Attributes:
        messages (list[BaseMessage]):
            What it contains:A list of message objects like:HumanMessage,
                                                            AIMessage,
                                                            SystemMessage

    Important:
        Annotated[..., add_messages] tells LangGraph:
        → "Append new messages instead of overwriting"( Every iteration appends a new AI Message)

        Without add_messages:
        ❌ You overwrite history

        With it:
        ✅ You build conversation memory
    """
    messages: Annotated[list[BaseMessage], add_messages] #This line defines BOTH:✅ what the graph expects as input ,✅ what the graph will return as output


    """
    🔹How input is actually given

    You pass input when you invoke the graph:

    graph.invoke({
        "messages": [HumanMessage(content="Hello")]
    })

    👉 So here:

    "messages" → input key
    value → list of BaseMessage
    
    🔹 How output is produced

    Every node in LangGraph returns a partial state like:

    return {
        "messages": [AIMessage(content="Hi!")]
    }
    """

# ==============================
# 3. NODE NAME CONSTANTS
# ==============================

REFLECT = "reflect"
GENERATE = "generate"


# ==============================
# 4. NODE DEFINITIONS
# ==============================

def generation_node(state: MessageGraph):
    """
    GENERATE NODE

    Purpose:
        Uses LLM to generate or improve content based on current messages.

    Steps:
        1. Reads messages from state
        2. Sends them to generate_chain
        3. Receives AI-generated response
        4. Returns it as a new message

    Input:
        state: MessageGraph (means current conversation history)
    Returns:
        dict → {"messages": [new_message]}

    Note:
        LangGraph automatically appends this to existing messages
        due to `add_messages`.
    """
    return {
        "messages": [
            generate_chain.invoke({"messages": state["messages"]})
        ]
    }


def reflection_node(state: MessageGraph):
    """
    REFLECT NODE

    Purpose:
        Critiques or improves the generated output.

    Steps:
        1. Sends messages to reflect_chain
        2. Gets feedback/improvement suggestion
        3. Wraps response as HumanMessage

    Why HumanMessage?
        So the next GENERATE step treats it as "user feedback"

    Returns:
        dict → {"messages": [HumanMessage(...)]}
    """
    res = reflect_chain.invoke({"messages": state["messages"]})

    return {
        "messages": [
            HumanMessage(content=res.content)
        ]
    }


# ==============================
# 5. GRAPH CONSTRUCTION
# ==============================

# Initialize graph with defined state schema
builder = StateGraph(state_schema=MessageGraph)

# Register nodes
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# Define entry point (where execution starts. So first flow will be from START node to GENERATE node)
builder.set_entry_point(GENERATE)


# ==============================
# 6. CONTROL FLOW LOGIC
# ==============================

def should_continue(state: MessageGraph):
    """
    Conditional routing function.

    Purpose:
        Decides whether to continue loop or stop.

    Logic:
        - If messages exceed 6 → END
        - Else → go to REFLECT node

    Why message count?
        Acts as a simple stopping condition to prevent infinite loop.
    """
    print("Checking condition...")
    print("Current message count:", len(state["messages"])) # Each full cycle does:GENERATE node → adds 1 AIMessage ,REFLECT node → adds 1 HumanMessage -each full iteration (generate + reflect) = +2
    #print("Current message :", state["messages"])
    print("=" * 40)

    if len(state["messages"]) > 6:
        print("Stopping graph")
        return END
    
    print("Continuing to REFLECT")
    return REFLECT


# Add conditional edge after GENERATE node
builder.add_conditional_edges(GENERATE, should_continue,path_map={END:END,REFLECT:REFLECT})

# Add loop: REFLECT → GENERATE
builder.add_edge(REFLECT, GENERATE)


# ==============================
# 7. COMPILE GRAPH
# ==============================

"""
Compiles the graph into an executable workflow.

After compilation:
    graph.invoke(input_state) can be used to run it
"""
graph = builder.compile()


# ==============================
# 8. DEBUG / VISUALIZATION
# ==============================

"""
These help you visualize the graph structure.
"""

# Mermaid diagram (can be pasted into excalidraw.com or mermaid.live )
print(graph.get_graph().draw_mermaid())

# ASCII diagram (terminal-friendly)
graph.get_graph().print_ascii()


# ==============================
# 9. EXECUTION
# ==============================

if __name__ == "__main__":
    print("Hello LangGraph")

    # Initial input state
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:
@LangChainAI
— newly Tool Calling feature is seriously underrated.

After a long wait, it's here - making the implementation of agents across different models with function calling super easy.

Made a video covering their newest blog post
"""
            )
        ]
    }

    """
    Execution Flow:

    1. GENERATE → improves tweet
    2. REFLECT → critiques it
    3. GENERATE → improves again
    4. Loop continues
    5. Stops when messages > 6
    """

    response = graph.invoke(inputs)

    # Final output (contains full message history) .LangGraph does NOT return just the last message.It returns the full state
    print(f" The response is : {response}")
    print("*"*60)
    print("*"*60)
    final_message = response["messages"][-1]

    print("\nFinal Output:\n", final_message.content)


# ==============================
# 🔁 SUMMARY OF FLOW
# ==============================

"""
Graph Structure:

    GENERATE
        ↓
    (Decision: should_continue)
       / \
     END  REFLECT
              ↓
          GENERATE (loop)

Key Concepts:

- State = shared memory (messages)
- Nodes = functions (generate, reflect)
- Edges = control flow
- Loop = iterative improvement (agent behavior)
"""

"""
LangGraph Step-by-Step Instructions (Point Format)
🔹 1. Define the Goal
Build an AI workflow that:
Generates output
Reflects on it
Improves it in a loop
This is an agentic loop (Generate → Reflect → Repeat)
🔹 2. Create the State (Memory)
Define a shared state using TypedDict
Store all messages in a list
Use add_messages to append instead of overwrite

👉 Key idea:

State = memory of the system
Every node reads and updates it
🔹 3. Import Required Components
Import:
StateGraph → builds workflow
END → stops execution
BaseMessage, HumanMessage → message types
Annotated → add metadata
Import your chains:
generate_chain
reflect_chain
🔹 4. Define Node Names
Create constants:
"generate"
"reflect"
These act as identifiers for nodes
🔹 5. Create GENERATE Node
Input: state (messages)
Action:
Call generate_chain
Pass all messages
Output:
New AI message added to state

👉 Purpose:

Improve or generate content
🔹 6. Create REFLECT Node
Input: state
Action:
Call reflect_chain
Analyze previous output
Convert output to HumanMessage

👉 Important Trick:

Treat AI feedback as human feedback
Helps LLM improve better
🔹 7. Initialize the Graph
Create graph using:
StateGraph(state_schema=MessageGraph)
This defines:
Structure of workflow
Structure of state
🔹 8. Add Nodes to Graph
Register nodes:
generate → generation_node
reflect → reflection_node
🔹 9. Set Entry Point
Start execution at:
generate node

👉 Flow always begins here

🔹 10. Define Control Logic
Create function should_continue
Input: state
Output:
"reflect" → continue loop
END → stop

👉 Example logic:

If messages > 6 → stop
Else → reflect
🔹 11. Add Conditional Edge
From generate:
Use should_continue to decide next step

👉 This controls:

Loop vs exit
🔹 12. Add Loop Edge
From reflect → generate

👉 This creates:

Iterative improvement loop
🔹 13. Compile the Graph
Call:
builder.compile()

👉 Converts design into executable workflow

🔹 14. Visualize the Graph
Use:
draw_mermaid() → diagram
print_ascii() → terminal view

👉 Helps debug and understand flow

🔹 15. Prepare Input
Create initial state:
List with one HumanMessage

👉 Example:

“Improve this tweet”
🔹 16. Execute the Graph
Call:
graph.invoke(inputs)
🔹 17. Understand Execution Flow
Start at GENERATE
Generate improved output
Check condition
If continue → go to REFLECT
Reflect on output
Go back to GENERATE
Repeat loop
Stop when condition met
🔹 18. Key Concepts to Remember
State → shared memory
Node → function (step)
Edge → connection (flow)
Conditional Edge → decision logic
Reducer (add_messages) → how state updates
Loop → enables agent behavior
🔹 19. Important Insight
Each iteration sees:
Original input
Previous outputs
Feedback

👉 This gives context-aware improvement

🔹 20. What You Built
A self-improving AI agent
With:
Memory ✔
Feedback loop ✔
Control flow ✔
Iteration ✔    
"""