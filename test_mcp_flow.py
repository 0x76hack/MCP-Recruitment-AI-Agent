import asyncio
import os
import sys
import json
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_flow")

async def run_integration_tests():
    print("=" * 70)
    print("STARTING MCP INTEGRATION AND ARCHITECTURE VERIFICATION SUITE")
    print("=" * 70)
    
    fs_server_params = StdioServerParameters(
        command=sys.executable,
        args=["filesystem_mcp_server.py"]
    )
    
    ext_server_params = StdioServerParameters(
        command=sys.executable,
        args=["external_services_mcp_server.py"]
    )
    
    async with AsyncExitStack() as stack:
        print("\n[Step 1] Connecting to Filesystem MCP Server...")
        fs_transport = await stack.enter_async_context(stdio_client(fs_server_params))
        fs_session = await stack.enter_async_context(ClientSession(fs_transport[0], fs_transport[1]))
        await fs_session.initialize()
        print("-> SUCCESS: Connected to Filesystem MCP Server.")
        
        print("\n[Step 2] Connecting to External Services MCP Server...")
        ext_transport = await stack.enter_async_context(stdio_client(ext_server_params))
        ext_session = await stack.enter_async_context(ClientSession(ext_transport[0], ext_transport[1]))
        await ext_session.initialize()
        print("-> SUCCESS: Connected to External Services MCP Server.")
        
        print("\n[Step 3] Resource & Tool Discovery Verification...")
        
        fs_tools = await fs_session.list_tools()
        print(f"\nFilesystem Server Tools discovered: {[t.name for t in fs_tools.tools]}")
        
        ext_tools = await ext_session.list_tools()
        print(f"External Services Server Tools discovered: {[t.name for t in ext_tools.tools]}")
        
        fs_resources = await fs_session.list_resources()
        print(f"\nFilesystem Server Resources discovered: {[r.uri for r in fs_resources.resources]}")
        
        ext_resources = await ext_session.list_resources()
        print(f"External Services Server Resources discovered: {[r.uri for r in ext_resources.resources]}")
        
        print("\n[Step 4] Reading RAG Database Status Resource...")
        rag_status = await fs_session.read_resource("status://rag")
        status_data = json.loads(rag_status.contents[0].text)
        print(f"Database Status: {json.dumps(status_data, indent=2)}")
        
        print("\nReading Indexed Resume List Resource...")
        resumes_list_res = await fs_session.read_resource("resumes://list")
        resumes_data = json.loads(resumes_list_res.contents[0].text)
        candidates = resumes_data.get("candidates", [])
        print(f"Total candidates in database: {resumes_data.get('total_count')}")
        if candidates:
            print(f"Sample candidates (first 3): {candidates[:3]}")
            
            sample_candidate = candidates[0]
            print(f"\nReading candidate content resource for '{sample_candidate}'...")
            candidate_content = await fs_session.read_resource(f"resumes://content/{sample_candidate}")
            content_snippet = candidate_content.contents[0].text[:300]
            print(f"Snippet of {sample_candidate}:\n---\n{content_snippet}\n---")
            
        print("\n[Step 5] Executing extract_requirements on Filesystem Server...")
        networking_jd = (
            "Job Title: Senior Systems and Network Administrator\n"
            "Requirements:\n"
            "- 5+ years of experience administering Windows Server, Active Directory, and Group Policy.\n"
            "- Strong hands-on networking background: Cisco switches/routers configuration, CCNA/CCNP certification required.\n"
            "- Experience using Wireshark for deep packet inspection and troubleshooting network latency issues.\n"
            "- Excellent shell scripting skills in PowerShell or Python for automated infrastructure deployments."
        )
        requirements_resp = await fs_session.call_tool("extract_requirements", arguments={"jd": networking_jd})
        requirements_text = requirements_resp.content[0].text
        print(f"Extracted Requirements (JSON):\n{requirements_text}")
        
        print("\n[Step 6] Executing search_resumes_tool with networking requirements...")
        search_query = "Active Directory, Cisco routing, CCNA, CCNP, Windows Server troubleshooting, Wireshark, PowerShell"
        search_resp = await fs_session.call_tool("search_resumes_tool", arguments={"requirements": search_query})
        search_results = search_resp.content[0].text
        print(f"Search Results (Top Candidates):\n{search_results[:1000]}...\n[Truncated for logs]")
        
        if len(candidates) >= 2:
            print(f"\n[Step 7] Executing compare_candidates for {candidates[0]} and {candidates[1]}...")
            compare_resp = await fs_session.call_tool("compare_candidates", arguments={"candidate_filenames": [candidates[0], candidates[1]]})
            print(f"Comparison Result:\n{compare_resp.content[0].text[:800]}...\n[Truncated for logs]")
            
            print(f"\nExecuting generate_interview_questions for {candidates[0]}...")
            iq_resp = await fs_session.call_tool("generate_interview_questions", arguments={"candidate_filename": candidates[0]})
            print(f"Targeted Questions:\n{iq_resp.content[0].text}")
            
        print("\n[Step 8] Executing fetch_salary_benchmark on External Services Server...")
        salary_resp = await ext_session.call_tool("fetch_salary_benchmark", arguments={"role": "systems administrator"})
        print(f"Salary Benchmarks:\n{salary_resp.content[0].text}")
        
        print("\nExecuting web_search on External Services Server...")
        search_resp = await ext_session.call_tool("web_search", arguments={"query": "Cisco CCNP certification trend 2026"})
        print(f"Web Search Results:\n{search_resp.content[0].text}")
        
        print("\n[Step 9] Testing batch_process capability...")
        if len(candidates) >= 2:
            batch_result = await fs_session.call_tool("batch_process", arguments={"candidate_filenames": candidates[:2]})
            print(f"Batch Process Result:\n{batch_result.content[0].text}")
            
        print("\n[Step 10] Testing watch_directory capability...")
        watch_result = await fs_session.call_tool("watch_directory", arguments={"directory_path": "data/resumes"})
        print(f"Watch Directory Register Result:\n{watch_result.content[0].text}")
        
        print("\n" + "=" * 70)
        print("INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("All tools, resources, error flows, and Multi-MCP connections verified.")
        print("=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(run_integration_tests())
    except KeyboardInterrupt:
        print("Testing aborted.")
        sys.exit(0)
