"""
N.O.V.A. Agent Brain  —  backend/agent/graph.py
================================================
Written for:
  langgraph==1.1.2
  langchain==1.2.12
  langchain-core==1.2.19
  langchain-openai==1.1.11
"""

import asyncio
import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START

from tools.ragtool import rag_search
from tools.mock_integrations import create_jira_ticket, send_slack_welcome
from tools.config import get_logger

load_dotenv()
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  STATE
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list        # HumanMessage / AIMessage history
    persona: dict         # role, tech_stack, experience_level, team, gaps
    checklist: dict       # task_id -> {title, description, category, priority, status}
    rag_context: str      # retrieved context injected before answer_node
    tool_results: list    # tool call results this turn
    current_step: str     # which node last ran


# ─────────────────────────────────────────────────────────────
# 2.  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

_BASE = """You are N.O.V.A. (Networked Onboarding Virtual Assistant), an intelligent
onboarding assistant for a software company.

YOUR ONLY PURPOSE is to help new employees complete their onboarding journey.

STRICT RULES:
- Only answer questions related to onboarding, company tools, HR policies, team workflows,
  and the new hire's role setup.
- If asked anything else, redirect: "I'm here specifically to help with your onboarding!"
- Always prefer information from the provided knowledge-base context.
- Never invent tool names, links, credentials, or policies.
- Be warm, encouraging, and concise."""

PERSONA_PROMPT = f"""{_BASE}

TASK — PERSONA EXTRACTION:
Analyse the conversation and extract a structured developer persona.
Return ONLY valid JSON, no markdown fences, no extra text:
{{
  "role": "<job title>",
  "tech_stack": ["<language/framework>"],
  "experience_level": "<junior|mid|senior>",
  "team": "<team name or null>",
  "goals": ["<goals for first 30 days>"],
  "gaps": ["<unfamiliar tools or processes>"]
}}
Use null for any field not mentioned."""

CHECKLIST_PROMPT = f"""{_BASE}

TASK — CHECKLIST GENERATION:
Given the developer persona below, generate a personalised onboarding checklist.

Persona:
{{persona_json}}

Return ONLY valid JSON, no markdown fences:
{{
  "task_001": {{
    "title": "<short action title>",
    "description": "<one sentence why this matters>",
    "category": "<Setup|Access|Learning|Social|Process>",
    "priority": "<high|medium|low>",
    "status": "pending",
    "role_specific": true
  }}
}}

Rules:
- 8 to 12 tasks total.
- At least 3 must be role_specific from persona tech_stack and gaps.
- Always include: hardware setup, access credentials, team introduction,
  read architecture docs, first PR or first ticket.
- Order by priority (high first)."""

ANSWER_PROMPT = f"""{_BASE}

KNOWLEDGE BASE CONTEXT (from company docs — prefer this):
-----
{{rag_context}}
-----

USER CHECKLIST STATUS:
{{checklist_summary}}

Answer the user using the context above.
- Cite context naturally: "According to our HR policy, ..."
- If context is insufficient, say so and offer to escalate.
- If the user asks for a Jira ticket or Slack message, confirm the tool result."""


# ─────────────────────────────────────────────────────────────
# 3.  LLM
# ─────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_LLM", "gemini-3-flash-preview"),
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ─────────────────────────────────────────────────────────────
# 4.  NODES
# ─────────────────────────────────────────────────────────────

def extract_persona_node(state: AgentState) -> dict:
    if state.get("persona") and state["persona"].get("role"):
        logger.info("[NOVA] Persona already extracted, skipping.")
        return {"current_step": "extract_persona_skipped"}

    logger.info("[NOVA] Extracting developer persona...")
    llm = _llm(temperature=0.1)

    messages = [SystemMessage(content=PERSONA_PROMPT)] + list(state.get("messages", []))
    response = llm.invoke(messages)

    try:
        raw = response.content.strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        persona = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[NOVA] Persona parse failed: {response.content[:200]}")
        persona = {
            "role": "Unknown", "tech_stack": [],
            "experience_level": "unknown", "team": None,
            "goals": [], "gaps": []
        }

    logger.info(f"[NOVA] Persona: {persona.get('role')}")
    return {"persona": persona, "current_step": "extract_persona"}


def generate_checklist_node(state: AgentState) -> dict:
    if state.get("checklist"):
        logger.info("[NOVA] Checklist exists, skipping.")
        return {"current_step": "generate_checklist_skipped"}

    logger.info("[NOVA] Generating checklist...")
    llm = _llm(temperature=0.2)

    prompt = CHECKLIST_PROMPT.replace(
        "{persona_json}", json.dumps(state.get("persona", {}), indent=2)
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate my personalised onboarding checklist."),
    ])

    try:
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        checklist = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[NOVA] Checklist parse failed: {response.content[:200]}")
        checklist = {
            "task_001": {
                "title": "Complete hardware setup",
                "description": "Get your laptop provisioned.",
                "category": "Setup", "priority": "high",
                "status": "pending", "role_specific": False,
            }
        }

    logger.info(f"[NOVA] Checklist: {len(checklist)} tasks")
    return {"checklist": checklist, "current_step": "generate_checklist"}


def answer_node(state: AgentState) -> dict:
    logger.info("[NOVA] Answering...")
    llm = _llm(temperature=0.4)

    checklist = state.get("checklist", {})
    checklist_summary = "\n".join(
        f"  [{v.get('status','pending').upper()}] {v.get('title','')}"
        for v in checklist.values()
    ) if checklist else "No checklist yet."

    rag_context = state.get("rag_context") or "No context retrieved."

    system_content = (
        ANSWER_PROMPT
        .replace("{rag_context}", rag_context)
        .replace("{checklist_summary}", checklist_summary)
    )

    latest = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            latest = m.content.lower()
            break

    tool_results = list(state.get("tool_results", []))

    if any(kw in latest for kw in ["jira", "ticket", "raise ticket", "create ticket"]):
        role = state.get("persona", {}).get("role", "New Hire")
        try:
            result = asyncio.run(create_jira_ticket(role, "Onboarding Setup"))
            tool_results.append({"tool": "create_jira_ticket", "result": result})
            logger.info(f"[NOVA] Jira: {result}")
        except Exception as e:
            tool_results.append({"tool": "create_jira_ticket", "result": f"Error: {e}"})

    if any(kw in latest for kw in ["slack", "welcome message", "send message"]):
        persona = state.get("persona", {})
        team = persona.get("team") or "general"
        channel = f"#{team.lower().replace(' ', '-')}"
        try:
            result = asyncio.run(send_slack_welcome(persona.get("role", "New Hire"), channel))
            tool_results.append({"tool": "send_slack_welcome", "result": result})
            logger.info(f"[NOVA] Slack: {result}")
        except Exception as e:
            tool_results.append({"tool": "send_slack_welcome", "result": f"Error: {e}"})

    messages = [SystemMessage(content=system_content)] + list(state.get("messages", []))

    if tool_results:
        summary = "\n".join(f"Tool '{r['tool']}': {r['result']}" for r in tool_results)
        messages.append(HumanMessage(content=f"[TOOL RESULTS]\n{summary}"))

    response = llm.invoke(messages)
    updated_messages = list(state.get("messages", [])) + [response]

    return {
        "messages": updated_messages,
        "tool_results": tool_results,
        "current_step": "answer",
    }


# ─────────────────────────────────────────────────────────────
# 5.  ROUTING
# ─────────────────────────────────────────────────────────────

def _router(state: AgentState) -> str:
    if not state.get("persona") or not state["persona"].get("role"):
        return "extract_persona"
    if not state.get("checklist"):
        return "generate_checklist"
    return "answer"


# ─────────────────────────────────────────────────────────────
# 6.  GRAPH  (langgraph 1.1.2 API)
# ─────────────────────────────────────────────────────────────

def build_nova_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("extract_persona", extract_persona_node)
    workflow.add_node("generate_checklist", generate_checklist_node)
    workflow.add_node("answer", answer_node)

    workflow.add_conditional_edges(
        START,
        _router,
        {
            "extract_persona": "extract_persona",
            "generate_checklist": "generate_checklist",
            "answer": "answer",
        }
    )

    workflow.add_edge("extract_persona", "generate_checklist")
    workflow.add_edge("generate_checklist", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


nova_graph = build_nova_graph()


# ─────────────────────────────────────────────────────────────
# 7.  PUBLIC HELPER  (called by main.py)
# ─────────────────────────────────────────────────────────────

def run_nova_turn(user_message: str, session_state: dict | None = None) -> dict:
    if session_state is None:
        session_state = {
            "messages": [],
            "persona": {},
            "checklist": {},
            "rag_context": "",
            "tool_results": [],
            "current_step": "start",
        }

    session_state["messages"] = list(session_state.get("messages", [])) + [
        HumanMessage(content=user_message)
    ]

    try:
        rag_result = rag_search(user_message)
        session_state["rag_context"] = rag_result.get("context", "")
    except Exception as e:
        logger.warning(f"[NOVA] RAG failed: {e}")
        session_state["rag_context"] = ""

    session_state["tool_results"] = []

    return nova_graph.invoke(session_state)


# ─────────────────────────────────────────────────────────────
# 8.  SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY in your .env file first.")
    else:
        print("=== N.O.V.A. Smoke Test ===\n")

        state = run_nova_turn(
            "Hi! I'm Alex, a new Backend Engineer. "
            "I work with Python and FastAPI. I've never used Jira before."
        )
        print("[Persona]", json.dumps(state["persona"], indent=2))
        print(f"[Checklist] {len(state['checklist'])} tasks")
        print("[N.O.V.A.]", state["messages"][-1].content)

        print("\n--- Turn 2 ---")
        state = run_nova_turn(
            "Can you create a Jira ticket for my hardware setup?",
            session_state=state
        )
        print("[N.O.V.A.]", state["messages"][-1].content)
        print("[Tools]", state["tool_results"])
