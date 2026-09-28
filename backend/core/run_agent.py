"""
Manual CLI harness for exercising the full DevPilot AI agent flow
end-to-end from a terminal, without going through the REST API.

Useful for the "Test main agent flow" / "Full end-to-end testing"
checklist items when you want to watch the agent's raw LangGraph
output directly. For frontend/API integration, use
POST /api/agent/run/ and /api/agent/resume/ instead
(see core/agent_service.py and README.md).

Run with:
    python -m core.run_agent
"""

import os
import uuid

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings"
)

import django
django.setup()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agent import agent


if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": f"devpilot-cli-{uuid.uuid4()}"
        }
    }

    task = input("\nEnter DevPilot task: ").strip()

    print("\n==============================")
    print("DEV PILOT REAL AGENT TEST")
    print("==============================")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=task)
            ],
            "approval": "",
            "proposed_path": "",
            "proposed_content": "",
            "test_result": "",
            "debug_result": "",
            "final_result": "",
            "failed_file": ""
        },
        config
    )

    print("\n==============================")
    print("AGENT RESULT")
    print("==============================")

    print(result)

    while "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0]

        print("\n==============================")
        print("HUMAN APPROVAL REQUIRED")
        print("==============================")

        print(interrupt_data.value)

        decision = input(
            "\nType 'approve' or 'reject': "
        ).strip().lower()

        result = agent.invoke(
            Command(resume=decision),
            config
        )

        print("\n==============================")
        print("GRAPH RESUMED")
        print("==============================")

        print(result)

    print("\n==============================")
    print("DEV PILOT FINISHED")
    print("==============================")