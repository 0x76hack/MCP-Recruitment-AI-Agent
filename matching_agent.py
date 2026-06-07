import os
import re
import sys
import logging
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from tools import AGENT_TOOLS, RAG_READY

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not api_key:
    print("CRITICAL ERROR: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

try:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name
    )
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize LLM: {e}")
    sys.exit(1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot_node(state: AgentState):
    """Core reasoning engine node."""
    sys_prompt = SystemMessage(content=(
        "You are an AI Recruitment Assistant. You have access to a database of PDF resumes. "
        "Use your tools to extract job requirements, search for candidates, compare them, and generate interview questions. "
        "When a user asks to search for candidates, ALWAYS use the 'search_resumes_tool'. "
        "When you use a tool, explain the results clearly and professionally to the user. "
        "If the user changes their requirements, you must execute a new search."
        "Format your output in clean Markdown."
    ))
    
    messages = [sys_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_execution_node(state: AgentState):
    """Executes the specific Python function requested by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"\n[System: Executing background tool -> {tool_name}]")
        
        tool_func = next((t for t in AGENT_TOOLS if t.name == tool_name), None)
        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            except Exception as e:
                error_msg = f"Execution failed for {tool_name}: {str(e)}"
                tool_responses.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                
    return {"messages": tool_responses}

workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_execution_node)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot", 
    should_continue, 
    {"tools": "tools", "__end__": END}
)
workflow.add_edge("tools", "chatbot")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def sanitize_input(user_input: str) -> str:
    """Removes non-printable characters and strips whitespace."""
    if not user_input:
        return ""
    return re.sub(r'[\x00-\x1F\x7F]', '', user_input).strip()

def run_cli():
    """Main execution loop for the terminal interface."""
    if not RAG_READY:
        print("\nInitialization failed. Exiting application. Please check logs.")
        return

    print("\n" + "="*60)
    print("Recruitment AI Agent Initialized.")
    print("Type 'quit' or 'exit' to terminate the session.")
    print("="*60 + "\n")
    
    config = {"configurable": {"thread_id": "cli_session_production"}}
    
    while True:
        try:
            raw_input = input("User: ")
            clean_input = sanitize_input(raw_input)
            
            if clean_input.lower() in ['quit', 'exit', 'q']:
                print("Terminating session. Goodbye.")
                break
                
            if not clean_input:
                continue
            
            for event in app.stream({"messages": [HumanMessage(content=clean_input)]}, config=config, stream_mode="values"):
                last_msg = event["messages"][-1]
                
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(f"\nAgent:\n{last_msg.content}\n")
                    print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected system error occurred: {e}\n")

if __name__ == "__main__":
    run_cli()