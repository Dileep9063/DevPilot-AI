import os
import atexit
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings"
)

import django
django.setup()


from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    SystemMessage
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


from core.tools import (
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    git_diff,
    write_file,
    run_tests,
    set_workspace,
    reset_workspace
)


# ============================================================
# 1. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    approval: str

    proposed_path: str

    proposed_content: str

    test_result: str

    debug_result: str

    final_result: str

    failed_file: str

    workspace_path: str


# ============================================================
# 2. LLM
# ============================================================

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

hf_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        max_new_tokens=1000,
    )
)


# ============================================================
# 3. LLM FALLBACK
# ============================================================

def is_transient_provider_error(exc: Exception) -> bool:
    """Return True for provider availability/rate-limit errors."""

    message = str(exc).lower()

    transient_markers = (
        "429",
        "rate limit",
        "too many requests",
        "503",
        "service unavailable",
        "unavailable",
        "temporarily unavailable",
        "high demand",
    )

    return any(marker in message for marker in transient_markers)


def invoke_llm(messages, tools=None):
    """
    Invoke Gemini first and automatically fall back to Hugging Face
    for transient provider availability/rate-limit failures.
    """

    primary = gemini_llm.bind_tools(tools) if tools else gemini_llm

    try:
        return primary.invoke(messages)
    except Exception as exc:
        if not is_transient_provider_error(exc):
            raise

        fallback = hf_llm.bind_tools(tools) if tools else hf_llm
        return fallback.invoke(messages)


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are DevPilot AI, an AI software engineering agent.

Your job is to understand software development tasks and
work with the provided codebase tools.

Available AI tools:

1. list_dir
   Use this to understand the project structure.

2. search_code
   Use this to find relevant code or text.

3. read_file
   Use this to inspect the contents of a specific file.

4. propose_file_change
   Use this when a code change is needed.
   Never directly modify files.

5. git_diff
   Use this to inspect the exact proposed changes.

Important rules:

- Start by understanding the project structure when necessary.
- Search for relevant code instead of making random searches.
- Read relevant files before proposing modifications.
- Do not repeatedly search for single letters or common characters.
- Do not call a tool if the information you already have is sufficient.
- Never directly modify files.
- All file modifications require human approval.
"""


# ============================================================
# 4. MAIN AGENT NODE
# ============================================================

def agent_node(state: AgentState):
    workspace_token = set_workspace(state.get("workspace_path", "workspace"))
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = invoke_llm(
            messages,
            tools=[
                search_code,
                read_file,
                list_dir,
                propose_file_change,
                git_diff,
            ],
        )
        return {"messages": [response]}
    finally:
        reset_workspace(workspace_token)


# ============================================================
# 5. TOOL NODE
# ============================================================

tool_executor = ToolNode([
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    git_diff,
])

def tool_node(state: AgentState):
    workspace_token = set_workspace(state.get("workspace_path", "workspace"))
    try:
        return tool_executor.invoke(state)
    finally:
        reset_workspace(workspace_token)


# ============================================================
# 6. TOOL ROUTING
# ============================================================

def route_after_tools(state: AgentState):
    """Route proposal results to extraction; otherwise continue analysis."""
    last_message = state["messages"][-1]
    if last_message.name == "propose_file_change":
        return "extract_proposal"
    return "agent"


# ============================================================
# 7. EXTRACT FILE PROPOSAL
# ============================================================

def extract_proposal_node(state: AgentState):

    last_message = state["messages"][-1]

    content = last_message.content

    if not content:
        return {}

    import ast

    try:
        result = ast.literal_eval(content)

    except (ValueError, SyntaxError):
        return {}

    if not isinstance(result, dict):
        return {}

    if result.get("status") != "pending_approval":
        return {}

    proposed_path = str(result.get("path", "")).strip()
    proposed_content = result.get("proposed_content", "")

    # Never send an empty path to the approval/write stages.
    if not proposed_path or not isinstance(proposed_content, str):
        return {
            "proposed_path": "",
            "proposed_content": "",
            "final_result": "Invalid file-change proposal: a non-empty file path is required."
        }

    return {
        "proposed_path": proposed_path,
        "proposed_content": proposed_content
    }


# ============================================================
# 8. AGENT ROUTING
# ============================================================

def should_continue(state: AgentState):
    """Only enter HITL when a valid proposal exists."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"

    if state.get("proposed_path") and state.get("proposed_content"):
        return "approval"

    # The model must keep working until it produces a concrete proposal.
    return "agent"


# ============================================================
# 9. HUMAN APPROVAL NODE
# ============================================================

def approval_node(state: AgentState):

    if not state.get("proposed_path") or not isinstance(state.get("proposed_content"), str):
        return {"approval": "reject"}

    # If this node is being resumed after HITL,
    # the approval value has already been supplied.
    if state.get("approval") in ["approve", "reject"]:
        return {
            "approval": state["approval"]
        }

    proposed_path = state.get(
        "proposed_path",
        ""
    )

    proposed_content = state.get(
        "proposed_content",
        ""
    )

    decision = interrupt({
        "question": "Do you approve this code change?",

        "message": (
            "DevPilot AI is requesting approval "
            "before modifying files."
        ),

        "path": proposed_path,

        "content": proposed_content
    })

    return {
        "approval": decision
    }

# ============================================================
# 10. APPROVAL ROUTING
# ============================================================

def route_after_approval(state: AgentState):

    if state["approval"] == "approve":
        return "write"

    return END


# ============================================================
# 11. WRITE NODE
# ============================================================

def write_node(state: AgentState):

    workspace_token = set_workspace(state.get("workspace_path", "workspace"))
    try:
        result = write_file.invoke({
        "path": state["proposed_path"],
        "content": state["proposed_content"]
    })

        return {
            "final_result": str(result),

            # Reset the consumed approval decision. Without this, a
        # second approval checkpoint later in the SAME run (e.g. the
        # AI-fix proposal after a failed test) would hit the "already
        # decided" branch in approval_node with the OLD decision and
        # skip asking a human again — silently auto-approving the
        # AI's fix.
            "approval": ""
        }
    finally:
        reset_workspace(workspace_token)


# ============================================================
# 12. TEST NODE
# ============================================================

def test_node(state: AgentState):
    workspace_token = set_workspace(state.get("workspace_path", "workspace"))
    try:
        result = run_tests.invoke({})

        return {
            "test_result": str(result)
        }
    finally:
        reset_workspace(workspace_token)


# ============================================================
# 13. TEST RESULT ROUTING
# ============================================================

def route_after_test(state: AgentState):

    test_result = state["test_result"]

    if "'status': 'passed'" in test_result:
        return "done"

    return "debug"


# ============================================================
# 14. DETERMINISTIC DEBUG ANALYZER
# ============================================================

def debug_node(state: AgentState):
    """
    Analyze pytest failure output without using the LLM.

    This node extracts the failed file and test.
    """

    test_result = state["test_result"]

    if "FAILED" not in test_result:

        return {
            "debug_result": "No test failure detected.",
            "failed_file": ""
        }

    import re

    match = re.search(
        r"FAILED\s+([^\s:]+)::([^\s-]+)",
        test_result
    )

    if match:

        failed_file = match.group(1)
        failed_test = match.group(2)

    else:

        failed_file = "Unknown"
        failed_test = "Unknown"

    debug_result = {
        "status": "failure_detected",
        "failed_file": failed_file,
        "failed_test": failed_test,
        "message": "Pytest reported a test failure."
    }

    return {
        "debug_result": str(debug_result),
        "failed_file": failed_file
    }


# ============================================================
# 15. AI DEBUG AGENT
# ============================================================

def ai_debug_node(state: AgentState):

    debug_result = state["debug_result"]
    test_result = state["test_result"]

    # Get the failing file directly from deterministic state.
    # Do not try to parse it from the AI's response.
    failed_file = state.get(
        "failed_file",
        ""
    )

    # Read the actual failing file before asking Gemini.
        # Read the failing test file.
    test_file_content = ""

    if failed_file:

        try:

            result = read_file.invoke({
                "path": failed_file
            })

            if isinstance(result, dict):

                test_file_content = result.get(
                    "content",
                    ""
                )

            else:

                test_file_content = str(result)

        except Exception as e:

            test_file_content = (
                f"Unable to read {failed_file}: {e}"
            )


    # Read the implementation file used by the test.
    implementation_content = ""

    try:

        result = read_file.invoke({
            "path": "auth.py"
        })

        if isinstance(result, dict):

            implementation_content = result.get(
                "content",
                ""
            )

        else:

            implementation_content = str(result)

    except Exception as e:

        implementation_content = (
            f"Unable to read auth.py: {e}"
        )
    prompt = f"""
You are the Debug Agent inside DevPilot AI.

A software test has failed.

Detected failure:
{debug_result}

Full pytest result:
{test_result}

Actual contents of the failing test file:
{test_file_content}

Actual contents of the implementation file:
{implementation_content}
Analyze the failure using the ACTUAL FILE CONTENT above.

Provide:

1. Root cause
2. Relevant file
3. Relevant test
4. Recommended code change

Important rules:

- Base your analysis on the actual file content.
- Do not invent functions, classes, variables, APIs, or files.
- Preserve the existing architecture.
- Do not modify any files.
- If the failing assertion is only a placeholder such as `assert False`,
  explicitly identify that.
- If more information is required, say what information is missing.

Be concise and technically specific.
"""

    response = invoke_llm(prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):

                text_parts.append(
                    item.get("text", "")
                )

        content = "\n".join(text_parts)

    return {
        "debug_result": content
    }


# ============================================================
# 16. DEBUG FIX PREPARATION
# ============================================================

def prepare_debug_fix_node(state: AgentState):

    debug_result = state["debug_result"]
    test_result = state["test_result"]

    # Get the failing file directly from state.
    failed_file = state.get(
        "failed_file",
        ""
    )

    # Read the actual file before proposing a fix.
    file_content = ""

    if failed_file:

        try:

            result = read_file.invoke({
                "path": failed_file
            })

            if isinstance(result, dict):

                file_content = result.get(
                    "content",
                    ""
                )

            else:

                file_content = str(result)

            # If the file was not found, try the workspace path.
            if "File does not exist" in file_content:

                result = read_file.invoke({
                    "path": f"workspace/{failed_file}"
                })

                if isinstance(result, dict):

                    file_content = result.get(
                        "content",
                        ""
                    )

                else:

                    file_content = str(result)

        except Exception as e:

            file_content = (
                f"Unable to read {failed_file}: {e}"
            )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    # Never replace a placeholder assertion with
    # meaningless code such as `assert True`.
    if "assert False" in file_content:

        return {
            "debug_result": (
                "The failing test contains a placeholder "
                "`assert False`. DevPilot cannot safely "
                "generate a meaningful replacement without "
                "knowing the intended login behavior."
            ),
            "proposed_path": "",
            "proposed_content": ""
        }

    # --------------------------------------------------------
    # FIX PROPOSAL PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the Fix Proposal Agent inside DevPilot AI.

A test has failed and another AI agent has already analyzed the failure.

AI Debug Diagnosis:
{debug_result}

Pytest Result:
{test_result}

Actual contents of the failing file:
{file_content}

Your task is to propose a concrete code fix.

Rules:

1. Do not modify any files.
2. Identify the exact file that should be changed.
3. Provide the complete proposed content of that file.
4. Preserve existing code that does not need to change.
5. Make the smallest reasonable change required to fix the failure.
6. Do not change the test merely to make the test pass unless the diagnosis clearly
   indicates that the test itself is incorrect.
7. Base the proposed fix on the actual file content.
8. Do not invent functions, classes, APIs, or modules that are not present.
9. Do not replace a placeholder assertion such as `assert False`
   with `assert True` or `pass`.
10. If there is insufficient information to create a meaningful fix,
    clearly state that instead of inventing code.

Return the result using exactly this format:

FILE: <exact file path>

CODE:
<complete proposed file content>
"""

    response = invoke_llm(prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):

                text_parts.append(
                    item.get("text", "")
                )

        content = "\n".join(text_parts)

    content = content.strip()

    # Make sure the expected format exists.
    if (
        "FILE:" not in content
        or "CODE:" not in content
    ):

        return {
            "debug_result": content,
            "proposed_path": "",
            "proposed_content": ""
        }

    # Separate FILE and CODE sections.
    file_part, code_part = content.split(
        "CODE:",
        1
    )

    proposed_path = file_part.replace(
        "FILE:",
        "",
        1
    ).strip()

    proposed_content = code_part.strip()

    # Remove markdown code fences if Gemini adds them.
    if proposed_content.startswith("```"):

        lines = proposed_content.splitlines()

        if (
            lines
            and lines[0].startswith("```")
        ):

            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):

            lines = lines[:-1]

        proposed_content = "\n".join(
            lines
        ).strip()

    return {
        "proposed_path": proposed_path,
        "proposed_content": proposed_content
    }


# ============================================================
# 17. DEBUG FIX ROUTING
# ============================================================

def route_after_debug_fix(state: AgentState):

    if (
        state["proposed_path"]
        and state["proposed_content"]
    ):

        return "approval"

    return END


# ============================================================
# 18. GRAPH BUILDER
# ============================================================

graph_builder = StateGraph(AgentState)


# ============================================================
# 19. REGISTER NODES
# ============================================================

graph_builder.add_node(
    "agent",
    agent_node
)

graph_builder.add_node(
    "tools",
    tool_node
)

graph_builder.add_node(
    "approval",
    approval_node
)

graph_builder.add_node(
    "write",
    write_node
)

graph_builder.add_node(
    "test",
    test_node
)

graph_builder.add_node(
    "debug",
    debug_node
)

graph_builder.add_node(
    "ai_debug",
    ai_debug_node
)

graph_builder.add_node(
    "prepare_debug_fix",
    prepare_debug_fix_node
)

graph_builder.add_node(
    "extract_proposal",
    extract_proposal_node
)


# ============================================================
# 20. START → AGENT
# ============================================================

graph_builder.add_edge(
    START,
    "agent"
)


# ============================================================
# 21. AGENT → TOOLS / APPROVAL
# ============================================================

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "approval": "approval"
    }
)


# ============================================================
# 22. TOOLS → AGENT / EXTRACT PROPOSAL
# ============================================================

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent": "agent",
        "extract_proposal": "extract_proposal"
    }
)


# ============================================================
# 23. EXTRACT PROPOSAL → APPROVAL / AGENT
# ============================================================

def route_after_extract_proposal(state: AgentState):
    if state.get("proposed_path") and isinstance(state.get("proposed_content"), str):
        return "approval"
    return "agent"

graph_builder.add_conditional_edges(
    "extract_proposal",
    route_after_extract_proposal,
    {
        "approval": "approval",
        "agent": "agent",
    }
)


# ============================================================
# 24. APPROVAL → WRITE / END
# ============================================================

graph_builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "write": "write",
        END: END
    }
)


# ============================================================
# 25. WRITE → TEST
# ============================================================

graph_builder.add_edge(
    "write",
    "test"
)


# ============================================================
# 26. TEST → DONE / DEBUG
# ============================================================

graph_builder.add_conditional_edges(
    "test",
    route_after_test,
    {
        "done": END,
        "debug": "debug"
    }
)


# ============================================================
# 27. DEBUG → AI DEBUG
# ============================================================

graph_builder.add_edge(
    "debug",
    "ai_debug"
)


# ============================================================
# 28. AI DEBUG → PREPARE FIX
# ============================================================

graph_builder.add_edge(
    "ai_debug",
    "prepare_debug_fix"
)


# ============================================================
# 29. PREPARE FIX → APPROVAL / END
# ============================================================

graph_builder.add_conditional_edges(
    "prepare_debug_fix",
    route_after_debug_fix,
    {
        "approval": "approval",
        END: END
    }
)


# ============================================================
# 30. CHECKPOINTER
# ============================================================

# ============================================================
# 30. CHECKPOINTER
# ============================================================

def create_checkpointer():
    """
    Create a persistent PostgreSQL-backed LangGraph checkpoint store.
    """

    uri = os.environ["LANGGRAPH_POSTGRES_URI"]

    pool = ConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )

    checkpointer = PostgresSaver(pool)

    return pool, checkpointer


checkpoint_pool, checkpointer = create_checkpointer()
atexit.register(checkpoint_pool.close)

# ============================================================
# 31. COMPILE GRAPH
# ============================================================

agent = graph_builder.compile(
    checkpointer=checkpointer
)

# NOTE: A single unified graph handles the entire flow:
#   agent -> tools -> approval (HITL) -> write -> test
#     -> (pass) END
#     -> (fail) debug -> ai_debug -> prepare_debug_fix -> approval (HITL for AI fix) -> write -> test -> ...
# The write/test/debug loop repeats automatically until tests pass,
# the AI gives up (no proposal), or a human rejects a proposed fix.
# There is intentionally no separate "debug_fix_graph" anymore — it was
# a duplicate of this same loop kept only for isolated manual testing.

# ============================================================
# 32. DIRECT EXECUTION TEST
# ============================================================

if __name__ == "__main__":

    print(
        "DevPilot AI agent loaded successfully."
    )