
import asyncio
import httpx
from langgraph.graph import StateGraph
from pydantic import BaseModel

class BookWritingState(BaseModel):
    input: str
    plan: str = None
    draft: str = None
    edited: str = None
    fact_checked: str = None
    final: str = None

def query_ollama(prompt):
    url = "http://localhost:8000/chat"
    payload = {
        "model": "gemma:2b",
        "messages": [{"role": "system", "content": prompt}]
    }
    response = httpx.post(url, json=payload,timeout=600)
    return response.json()["choices"][0]["message"]["content"]


def planning_agent(state):
    prompt = "Develop the book's concept, outline, characters, and world.along with a clear summery."
    response = query_ollama(prompt + "\n" + state.input)
    with open("plan.txt", "w", encoding="utf-8") as file:
        file.write(response)
    return {"plan": response}

def writing_agent(state):
    prompt = "Write detailed chapters based on the provided outline.1200 words each chapter"
    response = query_ollama(prompt + "\n" + state.plan)
    with open("write.txt", "w", encoding="utf-8") as file:
        file.write(response)
    return {"draft": response}

def editing_agent(state):
    prompt = "Edit the written chapters for clarity and coherence."
    response = query_ollama(prompt + "\n" + state.draft)
    print(f"editing agent :{response}")
    with open("edited.txt", "w", encoding="utf-8") as file:
        file.write(response)
    return {"edited": response}

def fact_checking_agent(state):
    prompt = "Verify the accuracy of all factual information."
    response = query_ollama(prompt + "\n" + state.edited)
    with open("factual.txt", "w", encoding="utf-8") as file:
        file.write(response)
    return {"fact_checked": response}

def publishing_agent(state):
    prompt = "Format the manuscript and prepare it for publication.simple"
    response = query_ollama(prompt + "\n" + state.fact_checked)
    with open("publish.txt", "w", encoding="utf-8") as file:
        file.write(response)
    return {"final": response}

graph = StateGraph(BookWritingState)
graph.add_node("Planning", planning_agent)
graph.add_node("Writing", writing_agent)
graph.add_node("Editing", editing_agent)
graph.add_node("Fact-Checking", fact_checking_agent)
graph.add_node("Publishing", publishing_agent)


graph.set_entry_point("Planning")
graph.add_edge("Planning", "Writing")
graph.add_edge("Writing", "Editing")
graph.add_edge("Editing", "Fact-Checking")
graph.add_edge("Fact-Checking", "Publishing")

workflow = graph.compile()

async def run_workflow(user_input):
    async for state in workflow.astream({"input": user_input}):
        print(state)
    print("Final Manuscript:", state.final)

if __name__ == "__main__":
    user_input = "Create a novel about an enginner trying to understand universe and love..5 chapters,each chapter 1200 words"
    asyncio.run(run_workflow(user_input))
