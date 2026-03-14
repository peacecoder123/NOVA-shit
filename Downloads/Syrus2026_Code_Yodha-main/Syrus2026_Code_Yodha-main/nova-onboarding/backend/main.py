"""
N.O.V.A. FastAPI Backend  —  backend/main.py
============================================
Changes vs original:
  - WebSocket /ws/chat now routes through the real LangGraph agent (run_nova_turn)
  - Session state is kept in memory per websocket connection
  - Sends structured JSON events: {type, content, checklist, persona, step}
  - /search endpoint unchanged (still calls rag_search directly)
"""

import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query

from tools.config import get_logger, Config
from tools.ragtool import rag_search
from agent.graph import run_nova_turn   # ← the only new import

logger = get_logger(__name__)

app = FastAPI(
    title="N.O.V.A. Onboarding API",
    description="Autonomous Developer Onboarding Agent"
)


@app.get("/")
async def read_root():
    return {"status": "N.O.V.A. Core is online.", "version": "2.0.0"}


@app.get("/search")
async def search_docs(query: str = Query(..., min_length=1)):
    logger.info(f"API Search: '{query}'")
    try:
        result = rag_search(query)
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Each websocket connection gets its own session state
    session_state = None

    await websocket.send_text(json.dumps({
        "type": "system",
        "content": "N.O.V.A. online. Tell me about yourself — your role and tech stack."
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            logger.debug(f"WS received: {raw}")

            try:
                payload = json.loads(raw)
                user_message = payload.get("message", raw)
            except json.JSONDecodeError:
                user_message = raw

            if not user_message.strip():
                continue

            # Send typing indicator to frontend
            await websocket.send_text(json.dumps({"type": "typing", "content": "..."}))

            try:
                # Run one turn through the LangGraph agent
                # run_nova_turn is synchronous (LangGraph 0.0.26 invoke is sync)
                session_state = run_nova_turn(
                    user_message=user_message,
                    session_state=session_state,
                )

                # Extract the AI reply
                ai_reply = session_state["messages"][-1].content

                # Send full structured response back to frontend
                await websocket.send_text(json.dumps({
                    "type": "message",
                    "content": ai_reply,
                    "persona": session_state.get("persona", {}),
                    "checklist": session_state.get("checklist", {}),
                    "step": session_state.get("current_step", ""),
                    "tool_results": session_state.get("tool_results", []),
                }))

            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"N.O.V.A. encountered an error: {str(e)}"
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket fatal error: {e}")


if __name__ == "__main__":
    logger.info("Starting N.O.V.A. v2.0")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
