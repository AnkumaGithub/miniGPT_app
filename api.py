from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
from typing import Optional
from enum import Enum

app = FastAPI()

class ModelType(str, Enum):
    CUSTOM = "custom"
    GPT2 = "gpt2"

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.3
    model_type: ModelType = ModelType.CUSTOM

def run_generation(request: GenerationRequest) -> str:
    try:
        script_map = {
            ModelType.CUSTOM: "gen.py",
            ModelType.GPT2: "inference.py"
        }

        result = subprocess.run(
            [
                "E:/PyCharm 2024.3.5/projects/.venv/Scripts/python.exe", script_map[request.model_type],
                "--prompt", request.prompt,
                "--max_tokens", str(request.max_tokens),
                "--temperature", str(request.temperature),
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[DEBUG] Script output: {result.stdout}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e.stderr}")
        return f"Error: {e.stderr}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    response = run_generation(request)

    if "Error" in response:
        raise HTTPException(
            status_code=500,
            detail=response
        )

    return {
        "prompt": request.prompt,
        "generated_text": response,
        "parameters": request.model_dump()
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)