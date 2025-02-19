from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.messages[-1]['content'] if request.messages else ""
        
        result = subprocess.run(
            ["ollama", "run", request.model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "choices": [{
                "message": {
                    "content": result.stdout.strip()
                }
            }]
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in model execution: {e.stderr.strip()}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Local Llama Models API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)