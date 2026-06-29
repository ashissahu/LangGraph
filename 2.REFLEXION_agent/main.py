#Langsmith Tracing : https://smith.langchain.com/public/2230b072-1bad-44cb-9b06-e07304a79d25/r/019f14b8-3cd8-7b30-9962-0e5ac4fc91a8
# Import typing utility to restrict return values of a function
from typing import Literal

# Core message types used in LangGraph / LangChain workflows
from langchain_core.messages import AIMessage, ToolMessage

# Graph building utilities
from langgraph.graph import END, START, StateGraph, MessagesState

# Your custom chains:
# - first_responder: generates initial answer
# - revisor: improves the answer using feedback/tools
from chains import revisor, first_responder

# Function that executes tool calls (e.g., search APIs)
from tool_executor import execute_tools


# Maximum number of iterations allowed (to avoid infinite loops)
MAX_ITERATIONS = 2


# ---------------------------
# NODE 1: Draft Initial Answer
# ---------------------------
def draft_node(state: MessagesState):
    """
    This node generates the FIRST response to the user's query.

    Input:
        state["messages"] -> list of conversation messages

    Process:
        Calls the 'first_responder' chain (LLM or prompt pipeline)

    Output:
        Returns updated state with new AI response appended
    """
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ---------------------------
# NODE 2: Revise Answer
# ---------------------------
def revise_node(state: MessagesState):
    """
    This node improves the answer after tool results are available.

    Input:
        state["messages"] -> includes:
            - original user query
            - initial draft
            - tool outputs

    Process:
        Calls 'revisor' chain which:
            - critiques previous answer
            - incorporates tool results
            - generates improved answer

    Output:
        Returns revised AI response
    """
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ---------------------------
# CONDITIONAL LOOP CONTROLLER
# ---------------------------
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """
    Controls whether the graph should:
        - Continue looping (execute tools again)
        - OR stop execution

    Logic:
        Count how many times tools were used (ToolMessage instances)

    Why?
        Each tool call = 1 iteration

    Stop Condition:
        If tool usage exceeds MAX_ITERATIONS → END

    Else:
        Continue execution → go back to tool execution
    """

    # Count number of tool messages in conversation
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )

    # Number of iterations equals tool usage count
    num_iterations = count_tool_visits

    # If exceeded max allowed iterations → stop
    if num_iterations > MAX_ITERATIONS:
        return END

    # Otherwise continue loop
    return "execute_tools"


# ---------------------------
# BUILD THE GRAPH
# ---------------------------

# Create a graph with message-based state
builder = StateGraph(MessagesState)

# Add nodes (steps in workflow)
builder.add_node("draft", draft_node)              # Step 1: initial answer
builder.add_node("execute_tools", execute_tools)   # Step 2: run tools (search, etc.)
builder.add_node("revise", revise_node)            # Step 3: refine answer

# Define flow of execution
builder.add_edge(START, "draft")                   # Start → draft
builder.add_edge("draft", "execute_tools")         # draft → tools
builder.add_edge("execute_tools", "revise")        # tools → revise

# Add loop:
# After revise, decide whether to:
#   - go back to tools
#   - or END
builder.add_conditional_edges(
    "revise",
    event_loop,
    ["execute_tools", END]
)

# Compile graph into executable object
graph = builder.compile()


# ---------------------------
# VISUALIZE GRAPH STRUCTURE
# ---------------------------

# Prints Mermaid diagram (helps visualize flow)
print(graph.get_graph().draw_mermaid())


# ---------------------------
# RUN THE GRAPH
# ---------------------------

# Invoke graph with user input
res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)


# ---------------------------
# EXTRACT FINAL ANSWER
# ---------------------------

# Get last message from conversation
last_message = res["messages"][-1]

# If final message contains tool calls, extract structured answer
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

# Print full state (for debugging / inspection)
print(res)

"""
High-Level Flow (Important)
User Query
   ↓
Draft Node (LLM generates initial answer)
   ↓
Execute Tools (search / APIs)
   ↓
Revise Node (improves answer using tool results)
   ↓
Loop Controller:
   ├── Continue → execute_tools
   └── Stop → END
"""