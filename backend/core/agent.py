import os
import atexit
import ast
import re

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings"
)

import django

django.setup()

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


from core.tools import (
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    propose_new_file,
    git_diff,
    git_status,
    create_agent_branch,
    write_file,
    write_new_file,
    run_tests,
    set_workspace,
    reset_workspace,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_AGENT_STEPS = 15
MAX_DEBUG_ATTEMPTS = 3


# ============================================================
# 2. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    # Run identity
    thread_id: str

    # Original user request
    task: str

    # Conversation/tool history
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # Human approval
    approval: str

    # Current proposed file change
    proposed_path: str
    proposed_content: str

    # Type of proposal
    # "existing_file" or "new_file"
    proposal_type: str

    # Git branch operation
    branch_result: str

    # Test/debug information
    test_result: str
    debug_result: str
    final_result: str

    # Failure information
    failed_file: str

    # Workspace used by this run
    workspace_path: str

    # Safety limits
    agent_steps: int
    debug_attempts: int


# ============================================================
# 3. LLM PROVIDERS
# ============================================================

# Gemini
#
# Kept available for future provider switching/fallback.

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# Hugging Face
#
# Kept available for future provider switching/fallback.

hf_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        max_new_tokens=1000,
    )
)


# OpenRouter

openrouter_llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


# Current primary provider

PRIMARY_LLM = openrouter_llm


# ============================================================
# 4. PROVIDER ERROR DETECTION
# ============================================================

def is_transient_provider_error(exc: Exception) -> bool:
    """
    Return True when the provider appears temporarily unavailable,
    rate-limited, or overloaded.
    """

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
        "timeout",
    )

    return any(
        marker in message
        for marker in transient_markers
    )


# ============================================================
# 5. LLM INVOCATION
# ============================================================

def invoke_llm(messages, tools=None):
    """
    Invoke the current primary LLM.

    OpenRouter is currently used as the primary provider because
    Gemini free-tier quota may be exhausted.

    Provider fallback can be added later without changing the
    LangGraph workflow.
    """

    llm = PRIMARY_LLM

    if tools:
        llm = llm.bind_tools(tools)

    try:

        return llm.invoke(messages)

    except Exception as exc:

        print(
            "LLM ERROR:",
            repr(exc)
        )

        raise


# ============================================================
# 6. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are DevPilot AI, an AI software engineering agent.

Your job is to understand the user's software-development task,
inspect the provided repository, propose the smallest correct
code change, obtain human approval, and then allow the system
to apply the approved change.

============================================================
AVAILABLE TOOLS
============================================================

1. list_dir

Use this to understand the project structure.

Use it when you need to determine:
- framework
- programming language
- available files
- tests
- configuration
- project structure

Do not repeatedly list the same directory if nothing has changed.

------------------------------------------------------------

2. search_code

Use this to find relevant:
- files
- functions
- classes
- imports
- routes
- APIs
- configuration
- tests
- documentation
- symbols

Use targeted searches.

Do not repeatedly search the same term after the information
has already been established.

Do not perform random searches merely to find something to modify.

------------------------------------------------------------

3. read_file

Use this to inspect the exact contents of a relevant file.

Always read an existing file before proposing a modification.

------------------------------------------------------------

4. propose_file_change

Use this when an EXISTING file needs to be modified.

IMPORTANT:

- First read the target file.
- old_text must exactly match existing text.
- new_text must contain only the replacement section.
- Do not provide the entire file as old_text.
- Do not provide a unified diff.
- Do not directly modify files.
- The tool only creates a proposal.
- Human approval is required before writing.

------------------------------------------------------------

5. propose_new_file

Use this when the requested change requires creating a NEW file.

IMPORTANT:

- First determine that the requested file does not already exist.
- Use the repository's existing language/framework conventions.
- Do not silently change the requested language.
- Do not modify an unrelated existing file instead.
- Provide the intended file path.
- Provide the complete content of the new file.
- Do not directly create the file.
- The tool only creates a proposal.
- Human approval is required before the file is created.

Use propose_file_change for existing files.

Use propose_new_file for new files.

------------------------------------------------------------

6. git_diff

Use this to inspect proposed/current repository changes when useful.

------------------------------------------------------------

7. git_status

Use this to inspect the current Git working-tree status.

Do not modify the repository with git commands.

============================================================
TASK FIDELITY
============================================================

The user's request is the source of truth.

Preserve the user's requested:

- programming language
- framework
- file type
- filename
- feature
- behavior
- scope

NEVER silently change the user's request.

For example:

User:
"Add a hello() function to a Python file."

Repository:
React/Vite
No Python files

Correct behavior:

1. Verify that no suitable Python file exists.
2. Do not switch to JavaScript.
3. Do not modify App.jsx.
4. Do not invent an unrelated file.
5. Do not reinterpret the task.
6. If the user explicitly requested Python and no appropriate
   Python file exists, report that limitation.
7. Stop unless the available tools can safely create the
   specifically requested Python file.

Incorrect behavior:

- Add hello() to App.jsx.
- Add hello() to a JavaScript file.
- Convert the request into a React task.
- Modify an unrelated file.
- Keep searching indefinitely.

============================================================
LANGUAGE AND FRAMEWORK RULES
============================================================

If the user explicitly requests Python:

Only modify Python code/files unless the task explicitly requires
changes to another file.

If the user explicitly requests JavaScript:

Do not switch to Python.

If the user explicitly requests Django:

Do not replace the solution with Flask/FastAPI/Node/etc.

If the user explicitly requests React:

Do not replace it with another frontend framework.

The repository may contain multiple languages.

Use the language/framework relevant to the user's request.

If the user does not specify a language or framework:

Prefer the language/framework already used by the relevant
repository area.

============================================================
REPOSITORY INVESTIGATION
============================================================

Before making a change:

1. Understand the repository structure.
2. Identify the relevant language/framework.
3. Find the relevant file.
4. Read the relevant file.
5. Understand surrounding code.
6. Determine the smallest safe modification.

Once sufficient information has been gathered:

STOP INVESTIGATING.

Do not repeatedly search for the same information.

============================================================
WHEN THE REQUESTED FILE DOES NOT EXIST
============================================================

If the user asks to modify a file that does not exist:

First determine whether another existing file is clearly intended.

If no suitable file exists:

If the user explicitly requested a particular language/file type,
do not silently switch to another language.

If creating a new file directly satisfies the user's request,
use propose_new_file.

Do not modify an unrelated existing file.

============================================================
BEFORE PROPOSING A CHANGE
============================================================

Verify all of the following:

1. The target file is correct.
2. If modifying an existing file, the target file was read.
3. The requested language is correct.
4. The requested framework is correct.
5. The proposed change directly satisfies the task.
6. Existing unrelated functionality is preserved.
7. The change is actually different from the current file.
8. No unrelated functionality is being changed.

If any condition fails:

Do not propose the change.

============================================================
MINIMAL CHANGE PRINCIPLE
============================================================

Make the smallest reasonable change required by the user.

Do not:
- rewrite entire files unnecessarily
- refactor unrelated code
- rename unrelated variables
- modify unrelated tests
- change dependencies unnecessarily
- change architecture without permission

============================================================
GIT WORKFLOW
============================================================

DevPilot works inside an isolated cloned repository workspace.

The AI may inspect Git status and diffs.

The AI must NOT directly create branches, commit changes,
or push changes.

After human approval, the LangGraph workflow controls the
Git branch creation.

The branch is created before the approved file modification.

The current workflow does NOT push to GitHub yet.

============================================================
TESTS
============================================================

After an approved change:

- Run the repository's appropriate test/build command.
- Use the existing project's tooling.
- Do not assume pytest for a JavaScript repository.
- Do not assume npm for a Python repository.
- Use the repository structure/package configuration to determine
  the appropriate test command.

If tests pass:

Complete the run.

If tests fail:

Analyze the actual failure.

============================================================
DEBUGGING
============================================================

When tests fail:

1. Identify the actual failing test/file.
2. Read the actual failing file.
3. Use the actual test output.
4. Do not invent files.
5. Do not invent functions.
6. Do not invent APIs.
7. Do not blindly modify the failing test.
8. Preserve existing architecture.
9. Propose the smallest meaningful fix.

If the failure is caused by a placeholder test such as:

assert False

do not replace it with:

assert True

or:

pass

Instead, explain that the test does not contain enough meaningful
behavior to determine the intended implementation.

============================================================
HUMAN-IN-THE-LOOP
============================================================

The AI must never directly modify files.

Every file modification must go through:

AI proposal
    ↓
Human approval
    ↓
Git branch creation
    ↓
Write
    ↓
Test

AI-generated debug fixes must also require human approval.

Never reuse an old approval decision for a new proposal.

============================================================
STOP CONDITIONS
============================================================

Stop when:

1. The requested change cannot safely be completed.
2. Required information does not exist.
3. Available tools cannot perform the required operation.
4. The requested change has been completed and tests pass.
5. A human rejects a proposed change.
6. The investigation limit has been reached.
7. The debug retry limit has been reached.
8. Git branch creation fails.

Never loop indefinitely.
"""


# ============================================================
# 7. HELPER: EXTRACT MESSAGE CONTENT
# ============================================================

def message_text(content) -> str:
    """
    Normalize LangChain message content into plain text.
    """

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):

                parts.append(
                    item.get("text", "")
                )

            elif isinstance(item, str):

                parts.append(item)

        return "\n".join(parts)

    return str(content)


# ============================================================
# 8. MAIN AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        step_count = (
            state.get(
                "agent_steps",
                0
            )
            + 1
        )

        task = state.get(
            "task",
            ""
        )

        messages = [
            SystemMessage(
                content=(
                    SYSTEM_PROMPT
                    + "\n\n"
                    + "CURRENT USER TASK:\n"
                    + task
                )
            ),
            *state["messages"],
        ]

        response = invoke_llm(
            messages,
            tools=[
                search_code,
                read_file,
                list_dir,
                propose_file_change,
                propose_new_file,
                git_diff,
                git_status,
            ],
        )

        return {
            "messages": [response],
            "agent_steps": step_count,
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 9. TOOL NODE
# ============================================================

tool_executor = ToolNode([
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    propose_new_file,
    git_diff,
    git_status,
])


def tool_node(state: AgentState):

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        return tool_executor.invoke(
            state
        )

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 10. TOOL ROUTING
# ============================================================

def route_after_tools(state: AgentState):
    """
    Route proposal results to proposal extraction.

    All other tool results go back to the agent.
    """

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "name",
        ""
    ) in {
        "propose_file_change",
        "propose_new_file",
    }:

        return "extract_proposal"

    return "agent"


# ============================================================
# 11. EXTRACT FILE PROPOSAL
# ============================================================

def extract_proposal_node(
    state: AgentState
):

    last_message = state["messages"][-1]

    content = message_text(
        last_message.content
    )

    if not content:

        return {}

    try:

        result = ast.literal_eval(
            content
        )

    except (
        ValueError,
        SyntaxError
    ):

        return {
            "proposed_path": "",
            "proposed_content": "",
            "proposal_type": "",
            "final_result": (
                "Unable to parse the file-change proposal."
            )
        }

    if not isinstance(
        result,
        dict
    ):

        return {
            "proposed_path": "",
            "proposed_content": "",
            "proposal_type": "",
            "final_result": (
                "Invalid file-change proposal."
            )
        }

    proposal_status = result.get(
        "status"
    )

    if proposal_status not in {
        "pending_approval",
        "pending_new_file_approval",
    }:

        return {
            "proposed_path": "",
            "proposed_content": "",
            "proposal_type": "",
        }

    proposed_path = str(
        result.get(
            "path",
            ""
        )
    ).strip()

    proposed_content = result.get(
        "proposed_content",
        ""
    )

    if (
        not proposed_path
        or not isinstance(
            proposed_content,
            str
        )
    ):

        return {
            "proposed_path": "",
            "proposed_content": "",
            "proposal_type": "",
            "final_result": (
                "Invalid file-change proposal: "
                "a non-empty path and valid content are required."
            )
        }

    if proposal_status == "pending_new_file_approval":

        proposal_type = "new_file"

    else:

        proposal_type = "existing_file"

    return {
        "proposed_path": proposed_path,
        "proposed_content": proposed_content,
        "proposal_type": proposal_type,
    }


# ============================================================
# 12. AGENT ROUTING
# ============================================================

def should_continue(
    state: AgentState
):

    last_message = state["messages"][-1]

    # Tool calls have priority.
    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"

    # Valid proposal exists.
    if (
        state.get(
            "proposed_path"
        )
        and state.get(
            "proposed_content"
        )
    ):

        return "approval"

    # Prevent infinite investigation loops.
    if (
        state.get(
            "agent_steps",
            0
        )
        >= MAX_AGENT_STEPS
    ):

        return "stop"

    return "agent"


# ============================================================
# 13. HUMAN APPROVAL NODE
# ============================================================

def approval_node(
    state: AgentState
):

    proposed_path = state.get(
        "proposed_path",
        ""
    )

    proposed_content = state.get(
        "proposed_content",
        ""
    )

    if (
        not proposed_path
        or not isinstance(
            proposed_content,
            str
        )
    ):

        return {
            "approval": "reject"
        }

    # If the graph is resumed after interrupt(),
    # LangGraph supplies the new decision.
    #
    # The write node resets approval after consuming it,
    # so a later debug fix cannot accidentally reuse the
    # previous decision.

    if state.get(
        "approval"
    ) in (
        "approve",
        "reject"
    ):

        return {
            "approval": state["approval"]
        }

    decision = interrupt({

        "question":
            "Do you approve this code change?",

        "message":
            "DevPilot AI is requesting human approval "
            "before modifying files.",

        "path":
            proposed_path,

        "content":
            proposed_content,
    })

    return {
        "approval": decision
    }


# ============================================================
# 14. REJECTION NODE
# ============================================================

def rejected_node(
    state: AgentState
):

    return {
        "final_result":
            "Human rejected the proposed change. "
            "No files were modified.",

        "test_result":
            "",

        "debug_result":
            "",

        "branch_result":
            "",
    }


# ============================================================
# 15. APPROVAL ROUTING
# ============================================================

def route_after_approval(
    state: AgentState
):

    if state.get(
        "approval"
    ) == "approve":

        return "create_branch"

    return "rejected"


# ============================================================
# 16. CREATE GIT BRANCH
# ============================================================

def create_branch_node(
    state: AgentState
):
    """
    Create a dedicated Git branch for this DevPilot run.

    This operation is controlled by LangGraph.
    The LLM cannot directly invoke this operation.
    """

    thread_id = state.get(
        "thread_id",
        ""
    )

    workspace_path = state.get(
        "workspace_path",
        ""
    )

    if not thread_id:

        return {
            "branch_result":
                "error: thread ID is missing."
        }

    if not workspace_path:

        return {
            "branch_result":
                "error: workspace path is missing."
        }

    branch_name = (
        f"devpilot/{thread_id}"
    )

    workspace_token = set_workspace(
        workspace_path
    )

    try:

        result = create_agent_branch.invoke({
            "branch_name":
                branch_name
        })

        if not isinstance(
            result,
            dict
        ):

            return {
                "branch_result":
                    f"error: {result}"
            }

        if result.get(
            "status"
        ) != "success":

            return {
                "branch_result":
                    f"error: {result}"
            }

        return {
            "branch_result":
                f"success: {branch_name}"
        }

    except Exception as exc:

        return {
            "branch_result":
                f"error: {exc}"
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 17. BRANCH ROUTING
# ============================================================

def route_after_branch(
    state: AgentState
):

    branch_result = state.get(
        "branch_result",
        ""
    )

    if branch_result.startswith(
        "success:"
    ):

        return "write"

    return END


# ============================================================
# 18. WRITE NODE
# ============================================================

def write_node(
    state: AgentState
):

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        proposal_type = state.get(
            "proposal_type",
            "existing_file"
        )

        if proposal_type == "new_file":

            result = write_new_file.invoke({
                "path":
                    state["proposed_path"],

                "content":
                    state["proposed_content"],
            })

        else:

            result = write_file.invoke({
                "path":
                    state["proposed_path"],

                "content":
                    state["proposed_content"],
            })

        return {

            "final_result":
                str(result),

            # Consume the current approval decision.
            "approval":
                "",

            # Clear the consumed proposal.
            "proposed_path":
                "",

            "proposed_content":
                "",

            "proposal_type":
                "",
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 19. TEST NODE
# ============================================================

def test_node(
    state: AgentState
):

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        result = run_tests.invoke({})

        return {
            "test_result":
                str(result)
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 20. TEST RESULT ROUTING
# ============================================================

def route_after_test(
    state: AgentState
):

    test_result = state.get(
        "test_result",
        ""
    )

    if "'status': 'passed'" in test_result:

        return "done"

    return "debug"


# ============================================================
# 21. DETERMINISTIC DEBUG ANALYZER
# ============================================================

def debug_node(
    state: AgentState
):

    test_result = state.get(
        "test_result",
        ""
    )

    if "FAILED" not in test_result:

        return {
            "debug_result":
                "No test failure detected.",

            "failed_file":
                "",
        }

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
        "status":
            "failure_detected",

        "failed_file":
            failed_file,

        "failed_test":
            failed_test,

        "message":
            "Pytest reported a test failure.",
    }

    return {
        "debug_result":
            str(debug_result),

        "failed_file":
            failed_file,
    }


# ============================================================
# 22. AI DEBUG AGENT
# ============================================================

def ai_debug_node(
    state: AgentState
):

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        debug_result = state.get(
            "debug_result",
            ""
        )

        test_result = state.get(
            "test_result",
            ""
        )

        failed_file = state.get(
            "failed_file",
            ""
        )

        test_file_content = ""

        # ----------------------------------------------------
        # Read actual failing file
        # ----------------------------------------------------

        if (
            failed_file
            and failed_file != "Unknown"
        ):

            try:

                result = read_file.invoke({
                    "path":
                        failed_file
                })

                if isinstance(
                    result,
                    dict
                ):

                    test_file_content = result.get(
                        "content",
                        ""
                    )

                else:

                    test_file_content = str(
                        result
                    )

            except Exception as exc:

                test_file_content = (
                    f"Unable to read "
                    f"{failed_file}: {exc}"
                )

        # ----------------------------------------------------
        # Debug prompt
        # ----------------------------------------------------

        prompt = f"""
You are the Debug Agent inside DevPilot AI.

A software test has failed.

Detected failure:
{debug_result}

Full test result:
{test_result}

Actual contents of the failing file:
{test_file_content}

Analyze the failure using ONLY the actual information above.

Provide:

1. Root cause
2. Relevant file
3. Relevant test
4. Recommended code change

Important rules:

- Base the diagnosis on the actual test output.
- Base the diagnosis on actual file contents.
- Do not invent files.
- Do not invent functions.
- Do not invent classes.
- Do not invent APIs.
- Do not assume a file such as auth.py exists.
- Do not assume the repository is Python.
- Preserve the existing architecture.
- Do not modify files.
- If the failing assertion is a placeholder such as:
  assert False
  explicitly identify it.
- If the available information is insufficient,
  clearly state what information is missing.
- Do not recommend changing an unrelated file.

Be concise and technically specific.
"""

        response = invoke_llm(
            prompt
        )

        content = message_text(
            response.content
        )

        return {
            "debug_result":
                content
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 23. DEBUG FIX PREPARATION
# ============================================================

def prepare_debug_fix_node(
    state: AgentState
):

    current_attempts = (
        state.get(
            "debug_attempts",
            0
        )
    )

    if current_attempts >= MAX_DEBUG_ATTEMPTS:

        return {
            "debug_result": (
                "Maximum debug attempts reached. "
                "DevPilot stopped the automatic fix loop."
            ),

            "proposed_path":
                "",

            "proposed_content":
                "",

            "proposal_type":
                "",
        }

    workspace_token = set_workspace(
        state.get(
            "workspace_path",
            "workspace"
        )
    )

    try:

        debug_result = state.get(
            "debug_result",
            ""
        )

        test_result = state.get(
            "test_result",
            ""
        )

        failed_file = state.get(
            "failed_file",
            ""
        )

        file_content = ""

        # ----------------------------------------------------
        # Read actual failing file
        # ----------------------------------------------------

        if (
            failed_file
            and failed_file != "Unknown"
        ):

            try:

                result = read_file.invoke({
                    "path":
                        failed_file
                })

                if isinstance(
                    result,
                    dict
                ):

                    file_content = result.get(
                        "content",
                        ""
                    )

                else:

                    file_content = str(
                        result
                    )

            except Exception as exc:

                file_content = (
                    f"Unable to read "
                    f"{failed_file}: {exc}"
                )

        # ----------------------------------------------------
        # Placeholder test protection
        # ----------------------------------------------------

        if "assert False" in file_content:

            return {
                "debug_result": (
                    "The failing test contains a placeholder "
                    "`assert False`. DevPilot cannot safely "
                    "generate a meaningful replacement without "
                    "knowing the intended behavior."
                ),

                "proposed_path":
                    "",

                "proposed_content":
                    "",

                "proposal_type":
                    "",
            }

        # ----------------------------------------------------
        # Fix proposal prompt
        # ----------------------------------------------------

        prompt = f"""
You are the Fix Proposal Agent inside DevPilot AI.

A test has failed and another AI agent has diagnosed the failure.

AI Debug Diagnosis:
{debug_result}

Test Result:
{test_result}

Failing File:
{failed_file}

Actual contents of the failing file:
{file_content}

Your job is to determine whether a safe, concrete code fix can
be proposed.

IMPORTANT:

- Do not modify files.
- Do not invent files.
- Do not invent functions.
- Do not invent classes.
- Do not invent APIs.
- Do not assume a specific framework.
- Do not assume a specific implementation file.
- Do not replace placeholder assertions with meaningless code.
- Preserve existing functionality.
- Make the smallest reasonable change.
- Do not modify tests merely to make them pass.
- The proposed fix must target an existing file.
- If the actual implementation file is not known from the available
  evidence, do NOT invent one.
- If there is insufficient information to create a meaningful fix,
  clearly say so.

If a safe fix can be proposed, return EXACTLY:

FILE: <exact existing file path>

CODE:
<complete proposed content of that existing file>

If a safe fix cannot be proposed, return:

CANNOT_FIX:
<reason>

Do not use markdown code fences.
"""

        response = invoke_llm(
            prompt
        )

        content = message_text(
            response.content
        ).strip()

        # ----------------------------------------------------
        # Cannot-fix response
        # ----------------------------------------------------

        if content.startswith(
            "CANNOT_FIX:"
        ):

            return {
                "debug_result":
                    content,

                "proposed_path":
                    "",

                "proposed_content":
                    "",

                "proposal_type":
                    "",
            }

        # ----------------------------------------------------
        # Validate expected response
        # ----------------------------------------------------

        if (
            "FILE:" not in content
            or "CODE:" not in content
        ):

            return {
                "debug_result":
                    content,

                "proposed_path":
                    "",

                "proposed_content":
                    "",

                "proposal_type":
                    "",
            }

        # ----------------------------------------------------
        # Separate FILE and CODE
        # ----------------------------------------------------

        file_part, code_part = content.split(
            "CODE:",
            1
        )

        proposed_path = (
            file_part
            .replace(
                "FILE:",
                "",
                1
            )
            .strip()
        )

        proposed_content = (
            code_part
            .strip()
        )

        # ----------------------------------------------------
        # Remove accidental code fences
        # ----------------------------------------------------

        if proposed_content.startswith(
            "```"
        ):

            lines = (
                proposed_content
                .splitlines()
            )

            if (
                lines
                and lines[0].startswith(
                    "```"
                )
            ):

                lines = lines[1:]

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):

                lines = lines[:-1]

            proposed_content = (
                "\n".join(lines)
                .strip()
            )

        # ----------------------------------------------------
        # Validate proposal
        # ----------------------------------------------------

        if not proposed_path:

            return {
                "debug_result":
                    "Debug agent returned an empty file path.",

                "proposed_path":
                    "",

                "proposed_content":
                    "",

                "proposal_type":
                    "",
            }

        if not proposed_content:

            return {
                "debug_result":
                    "Debug agent returned empty file content.",

                "proposed_path":
                    "",

                "proposed_content":
                    "",

                "proposal_type":
                    "",
            }

        # ----------------------------------------------------
        # Increment debug attempt only when a proposal exists
        # ----------------------------------------------------

        return {
            "proposed_path":
                proposed_path,

            "proposed_content":
                proposed_content,

            # Debug fixes always target an existing file.
            "proposal_type":
                "existing_file",

            "debug_attempts":
                current_attempts + 1,
        }

    finally:

        reset_workspace(
            workspace_token
        )


# ============================================================
# 24. DEBUG FIX ROUTING
# ============================================================

def route_after_debug_fix(
    state: AgentState
):

    if (
        state.get(
            "proposed_path"
        )
        and state.get(
            "proposed_content"
        )
    ):

        return "approval"

    return END


# ============================================================
# 25. GRAPH BUILDER
# ============================================================

graph_builder = StateGraph(
    AgentState
)


# ============================================================
# 26. REGISTER NODES
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
    "rejected",
    rejected_node
)

graph_builder.add_node(
    "create_branch",
    create_branch_node
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
# 27. START → AGENT
# ============================================================

graph_builder.add_edge(
    START,
    "agent"
)


# ============================================================
# 28. AGENT → TOOLS / APPROVAL / STOP
# ============================================================

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":
            "tools",

        "approval":
            "approval",

        "agent":
            "agent",

        "stop":
            END,
    }
)


# ============================================================
# 29. TOOLS → AGENT / EXTRACT PROPOSAL
# ============================================================

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent":
            "agent",

        "extract_proposal":
            "extract_proposal",
    }
)


# ============================================================
# 30. EXTRACT PROPOSAL → APPROVAL / AGENT
# ============================================================

def route_after_extract_proposal(
    state: AgentState
):

    if (
        state.get(
            "proposed_path"
        )
        and isinstance(
            state.get(
                "proposed_content"
            ),
            str
        )
        and state.get(
            "proposed_content"
        )
    ):

        return "approval"

    return "agent"


graph_builder.add_conditional_edges(
    "extract_proposal",
    route_after_extract_proposal,
    {
        "approval":
            "approval",

        "agent":
            "agent",
    }
)


# ============================================================
# 31. APPROVAL → CREATE BRANCH / REJECTED
# ============================================================

graph_builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "create_branch":
            "create_branch",

        "rejected":
            "rejected",
    }
)


# ============================================================
# 32. CREATE BRANCH → WRITE / END
# ============================================================

graph_builder.add_conditional_edges(
    "create_branch",
    route_after_branch,
    {
        "write":
            "write",

        END:
            END,
    }
)


# ============================================================
# 33. WRITE → TEST
# ============================================================

graph_builder.add_edge(
    "write",
    "test"
)


# ============================================================
# 34. REJECTED → END
# ============================================================

graph_builder.add_edge(
    "rejected",
    END
)


# ============================================================
# 35. TEST → DONE / DEBUG
# ============================================================

graph_builder.add_conditional_edges(
    "test",
    route_after_test,
    {
        "done":
            END,

        "debug":
            "debug",
    }
)


# ============================================================
# 36. DEBUG → AI DEBUG
# ============================================================

graph_builder.add_edge(
    "debug",
    "ai_debug"
)


# ============================================================
# 37. AI DEBUG → PREPARE FIX
# ============================================================

graph_builder.add_edge(
    "ai_debug",
    "prepare_debug_fix"
)


# ============================================================
# 38. PREPARE FIX → APPROVAL / END
# ============================================================

graph_builder.add_conditional_edges(
    "prepare_debug_fix",
    route_after_debug_fix,
    {
        "approval":
            "approval",

        END:
            END,
    }
)


# ============================================================
# 39. CHECKPOINTER
# ============================================================

def create_checkpointer():
    """
    Create a persistent PostgreSQL-backed LangGraph
    checkpoint store.
    """

    uri = os.environ[
        "LANGGRAPH_POSTGRES_URI"
    ]

    pool = ConnectionPool(
        conninfo=uri,

        min_size=1,

        max_size=5,

        kwargs={
            "autocommit":
                True,

            "prepare_threshold":
                0,
        },
    )

    checkpointer = PostgresSaver(
        pool
    )

    return (
        pool,
        checkpointer
    )


checkpoint_pool, checkpointer = (
    create_checkpointer()
)


atexit.register(
    checkpoint_pool.close
)


# ============================================================
# 40. COMPILE GRAPH
# ============================================================

agent = graph_builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 41. GRAPH FLOW
# ============================================================

"""
DevPilot AI unified workflow:

                    ┌──────────────┐
                    │    AGENT     │
                    └──────┬───────┘
                           │
                    tool call?
                     /          \
                   yes           no
                   /              \
                TOOLS          proposal?
                 │             /       \
                 │           yes       no
                 │            /         \
                 └────────→ EXTRACT      AGENT
                              │
                              ▼
                           APPROVAL
                              │
                        approve/reject
                         /           \
                    reject          approve
                       │               │
                   REJECTED       CREATE BRANCH
                       │               │
                      END             WRITE
                                       │
                                      TEST
                                     /    \
                                  pass    fail
                                   │        │
                                  END      DEBUG
                                            │
                                         AI DEBUG
                                            │
                                      PREPARE FIX
                                            │
                                         proposal?
                                         /       \
                                       yes        no
                                        │          │
                                    APPROVAL      END
                                        │
                                  CREATE BRANCH
                                        │
                                      WRITE
                                        │
                                      TEST
                                        │
                                      repeat


Proposal types:

Existing file:
    propose_file_change
            ↓
    pending_approval
            ↓
       create branch
            ↓
       write_file()

New file:
    propose_new_file
            ↓
    pending_new_file_approval
            ↓
       create branch
            ↓
       write_new_file()


Git safety:

- Git status is available to the AI for inspection.
- Branch creation is controlled by LangGraph.
- The LLM cannot directly create branches.
- Main/master cannot be used by create_agent_branch().
- The branch is created only after human approval.
- Current version does NOT commit or push yet.


Safety limits:

- MAX_AGENT_STEPS = 15
- MAX_DEBUG_ATTEMPTS = 3

Human approval is required for every proposed file modification.
"""


# ============================================================
# 42. DIRECT EXECUTION TEST
# ============================================================

if __name__ == "__main__":

    print(
        "DevPilot AI agent loaded successfully."
    )