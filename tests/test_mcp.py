"""Tests for native Model Context Protocol (MCP) server."""

import io
import json

from hydrous import Memory
from hydrous.embeddings import TfidfEmbedder
from hydrous.llm import MockLLM
from hydrous.mcp import MCPServer, decode_rpc_messages, encode_rpc_message
from hydrous.schema import ABSTAIN_STRING


def _run_mcp_requests(server: MCPServer, requests: list[dict]) -> list[dict]:
    out = io.StringIO()
    server.stdout = out
    for req in requests:
        server.handle_message(json.dumps(req))
    return decode_rpc_messages(out.getvalue())


def test_mcp_initialize_and_ping(tmp_path):
    with Memory(
        store="local",
        store_path=str(tmp_path / "mcp.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    ) as m:
        server = MCPServer(m, default_user_id="alice")
        resps = _run_mcp_requests(
            server,
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            ],
        )
        assert len(resps) == 3
        # initialize response
        assert resps[0]["id"] == 1
        assert resps[0]["result"]["serverInfo"]["name"] == "hydrous-mcp"
        assert "tools" in resps[0]["result"]["capabilities"]

        # ping response
        assert resps[1]["id"] == 2
        assert resps[1]["result"] == {}

        # tools/list response
        assert resps[2]["id"] == 3
        tool_names = [t["name"] for t in resps[2]["result"]["tools"]]
        assert "memory_add" in tool_names
        assert "memory_search" in tool_names
        assert "memory_get_timeline" in tool_names
        assert "memory_ask" in tool_names
        assert "memory_handoff" in tool_names


def test_mcp_tools_flow_and_abstention(tmp_path):
    with Memory(
        store="local",
        store_path=str(tmp_path / "mcp_flow.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    ) as m:
        server = MCPServer(m, default_user_id="bob")

        # 1. memory_add
        resps = _run_mcp_requests(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_add",
                        "arguments": {"text": "I am building a web app in Svelte", "infer": False},
                    },
                }
            ],
        )
        assert resps[0]["id"] == 10
        assert resps[0]["result"]["isError"] is False
        add_result = json.loads(resps[0]["result"]["content"][0]["text"])
        assert add_result["results"][0]["event"] == "ADD"
        mid = add_result["results"][0]["id"]

        # 2. memory_search
        resps = _run_mcp_requests(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_search",
                        "arguments": {"query": "Svelte"},
                    },
                }
            ],
        )
        assert resps[0]["id"] == 11
        search_res = json.loads(resps[0]["result"]["content"][0]["text"])
        assert len(search_res["results"]) >= 1
        assert search_res["results"][0]["id"] == mid

        # 3. memory_get_timeline
        resps = _run_mcp_requests(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_get_timeline",
                        "arguments": {"memory_id": mid},
                    },
                }
            ],
        )
        assert resps[0]["id"] == 12
        timeline = json.loads(resps[0]["result"]["content"][0]["text"])
        assert isinstance(timeline, list)
        assert len(timeline) >= 1

        # 4. memory_handoff
        resps = _run_mcp_requests(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 13,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_handoff",
                        "arguments": {},
                    },
                }
            ],
        )
        assert resps[0]["id"] == 13
        handoff_text = resps[0]["result"]["content"][0]["text"]
        assert "Cross-Agent Handoff" in handoff_text
        assert "Svelte" in handoff_text

        # 5. memory_ask (abstention case)
        resps = _run_mcp_requests(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 14,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_ask",
                        "arguments": {"question": "What is my favorite animal?"},
                    },
                }
            ],
        )
        assert resps[0]["id"] == 14
        ask_text = resps[0]["result"]["content"][0]["text"]
        assert ABSTAIN_STRING in ask_text
        assert "Verified Absent in Memory" in ask_text


def test_mcp_resources_and_prompts(tmp_path):
    with Memory(
        store="local",
        store_path=str(tmp_path / "mcp_res.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    ) as m:
        server = MCPServer(m, default_user_id="charlie")
        m.add("We use PostgreSQL for the auth database", user_id="charlie", infer=False)

        resps = _run_mcp_requests(
            server,
            [
                {"jsonrpc": "2.0", "id": 20, "method": "resources/list"},
                {"jsonrpc": "2.0", "id": 21, "method": "resources/read", "params": {"uri": "hydrous://memory/charlie"}},
                {"jsonrpc": "2.0", "id": 22, "method": "prompts/list"},
                {"jsonrpc": "2.0", "id": 23, "method": "prompts/get", "params": {"name": "agent_handoff", "arguments": {"task_summary": "Refactor auth"}}},
            ],
        )
        assert resps[0]["id"] == 20
        assert len(resps[0]["result"]["resources"]) == 2

        assert resps[1]["id"] == 21
        body = json.loads(resps[1]["result"]["contents"][0]["text"])
        assert any("PostgreSQL" in r.get("memory", "") for r in body.get("results", []))

        assert resps[2]["id"] == 22
        assert resps[2]["result"]["prompts"][0]["name"] == "agent_handoff"

        assert resps[3]["id"] == 23
        prompt_msg = resps[3]["result"]["messages"][0]["content"]["text"]
        assert "Refactor auth" in prompt_msg
        assert "PostgreSQL" in prompt_msg


def test_mcp_stdio_content_length_framing(tmp_path):
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    framed = encode_rpc_message(ping)
    stdin = io.TextIOWrapper(io.BytesIO(framed), encoding="utf-8")
    stdout_buf = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_buf, encoding="utf-8", write_through=True)

    with Memory(
        store="local",
        store_path=str(tmp_path / "mcp_frame.sqlite3"),
        llm=MockLLM(),
        embedder=TfidfEmbedder(),
    ) as m:
        server = MCPServer(m, default_user_id="dana", stdin=stdin, stdout=stdout)
        server.serve_forever()

    stdout.flush()
    messages = decode_rpc_messages(stdout_buf.getvalue())
    assert len(messages) == 1
    assert messages[0]["id"] == 1
    assert messages[0]["result"] == {}
    assert stdout_buf.getvalue().startswith(b"Content-Length:")
