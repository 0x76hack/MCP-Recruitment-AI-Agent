import os
import re
import sys
import logging
import asyncio
import json
from typing import TypedDict, Annotated, List, Dict, Any
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("matching_agent")
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not api_key:
    print("CRITICAL SECURITY ERROR: OPENROUTER_API_KEY not found in environment or .env file.")
    sys.exit(1)

try:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name
    )
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize LLM: {e}")
    sys.exit(1)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


TOOL_TO_SESSION: Dict[str, ClientSession] = {}
llm_with_tools = None


async def chatbot_node(state: AgentState):
    """Core reasoning engine node that invokes LLM with bound MCP tools."""
    sys_prompt = SystemMessage(content=(
        "You are an AI Recruitment Assistant. You have access to a database of PDF resumes. "
        "Use your tools to extract job requirements, search for candidates, compare them, and generate interview questions. "
        "You also have access to external tools like web search and salary benchmarks. "
        "When a user asks to search for candidates, ALWAYS use the 'search_resumes_tool'. "
        "When you use a tool, explain the results clearly and professionally to the user. "
        "If the user changes their requirements, you must execute a new search."
        "Format your output in clean Markdown."
    ))
    
    messages = [sys_prompt] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def tool_execution_node(state: AgentState):
    """Executes the specific MCP tool requested by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"\n[System: Executing MCP Tool -> {tool_name}]")
        
        session = TOOL_TO_SESSION.get(tool_name)
        if session:
            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                
                content_str = ""
                if hasattr(result, "content") and result.content:
                    content_str = "\n".join([c.text for c in result.content if hasattr(c, "text")])
                else:
                    content_str = str(result)
                    
                tool_responses.append(ToolMessage(content=content_str, tool_call_id=tool_call["id"]))
            except Exception as e:
                error_msg = f"Execution failed for MCP tool {tool_name}: {str(e)}"
                logger.error(error_msg)
                tool_responses.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
        else:
            error_msg = f"Error: Tool '{tool_name}' is not registered on any connected MCP server."
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


async def initialize_mcp_clients(stack: AsyncExitStack):
    """Launches local MCP servers and initializes client connections."""
    global llm_with_tools, TOOL_TO_SESSION
    
    fs_server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__) or ".", "filesystem_mcp_server.py")]
    )
    
    ext_server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__) or ".", "external_services_mcp_server.py")]
    )
    
    print("Connecting to Filesystem MCP Server...")
    fs_transport = await stack.enter_async_context(stdio_client(fs_server_params))
    fs_session = await stack.enter_async_context(ClientSession(fs_transport[0], fs_transport[1]))
    await fs_session.initialize()
    print("Filesystem MCP Server Connected.")
    
    print("Connecting to External Services MCP Server...")
    ext_transport = await stack.enter_async_context(stdio_client(ext_server_params))
    ext_session = await stack.enter_async_context(ClientSession(ext_transport[0], ext_transport[1]))
    await ext_session.initialize()
    print("External Services MCP Server Connected.")
    
    fs_tools = await fs_session.list_tools()
    ext_tools = await ext_session.list_tools()
    
    langchain_tools = []
    
    for tool in fs_tools.tools:
        TOOL_TO_SESSION[tool.name] = fs_session
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        })
        
    for tool in ext_tools.tools:
        TOOL_TO_SESSION[tool.name] = ext_session
        langchain_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        })
        
    llm_with_tools = llm.bind_tools(langchain_tools)
    
    try:
        status_res = await fs_session.read_resource("status://rag")
        if status_res and hasattr(status_res, "contents") and status_res.contents:
            status_data = json.loads(status_res.contents[0].text)
            print("\n" + "="*50)
            print("MCP RAG DB Initialized:")
            print(f"  Candidate resumes loaded: {status_data.get('number_of_candidates')}")
            print(f"  Embedding model in use:   {status_data.get('embedding_model')}")
            print(f"  LLM model configured:    {status_data.get('llm_model')}")
            print("="*50 + "\n")
    except Exception as e:
         print(f"Note: RAG Status resource not available on startup: {e}")
         
    return fs_session, ext_session


async def run_cli():
    """Main execution loop for the terminal interface."""
    print("\nStarting Recruitment AI Agent Client with Multi-MCP Integration...")
    
    async with AsyncExitStack() as stack:
        try:
            fs_session, ext_session = await initialize_mcp_clients(stack)
        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to establish connection to MCP servers: {e}")
            return
            
        print("\n" + "="*60)
        print("Recruitment AI Agent Client Initialized.")
        print("Type 'quit' or 'exit' to terminate the session.")
        print("="*60 + "\n")
        
        config = {"configurable": {"thread_id": "cli_session_production"}}
        
        try:
            watch_resp = await fs_session.call_tool("watch_directory", arguments={"directory_path": "data/resumes"})
            if hasattr(watch_resp, "content") and watch_resp.content:
                print(f"[System Watcher: {watch_resp.content[0].text}]")
        except Exception as e:
            print(f"[System Watcher Warning: Could not register directory watcher: {e}]")
            
        while True:
            try:
                raw_input = input("User: ")
                clean_input = sanitize_input(raw_input)
                
                if clean_input.lower() in ['quit', 'exit', 'q']:
                    print("Terminating session. Goodbye.")
                    break
                    
                if not clean_input:
                    continue
                
                async for event in app.astream({"messages": [HumanMessage(content=clean_input)]}, config=config, stream_mode="values"):
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
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nProcess terminated.")
        sys.exit(0)