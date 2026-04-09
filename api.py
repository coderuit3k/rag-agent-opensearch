from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from uuid import uuid4
from main import app as rag_app, route_question
from langchain_core.messages import HumanMessage

app = FastAPI()

# Request schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


# Predict route
def predict_route(question: str):
    state = {"messages": [HumanMessage(content=question)]}
    return route_question(state)

# Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid4())

    # route
    route = predict_route(req.message)

    # run RAG
    result = rag_app.invoke(
        {"messages": [("user", req.message)]},
        {"configurable": {"thread_id": thread_id}}
    )

    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "route": route,
        "thread_id": thread_id
    }


# UI endpoint
@app.get("/", response_class=HTMLResponse)
def ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()