from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
from typing import Optional

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.3
    stop_token: Optional[str] = "[EOS]"

def run_generation(request: GenerationRequest) -> str:
    try:
        result = subprocess.run(
            [
                "python", "gen.py",
                "--prompt", request.prompt,
                "--max_tokens", str(request.max_tokens),
                "--temperature", str(request.temperature),
                "--stop_token", request.stop_token
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
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
        "parameters": request.dict()
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)