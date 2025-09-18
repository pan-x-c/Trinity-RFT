import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()


# Forward openAI requests to a model


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    url = await request.app.state.server.get_running_model_url()
    async with httpx.AsyncClient() as client:
        print(f"Forwarding request to {url}/v1/chat/completions")
        print(f"Extra json: {body.get('task_id', {})}")
        resp = await client.post(f"{url}/v1/chat/completions", json=body)
    return resp.json()


@app.get("/health")
async def health(request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


async def serve_http(app: FastAPI, port: int = None, localmode: bool = False):
    host = "localhost" if localmode else "0.0.0.0"
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def run_app(server, port: int = None, localmode: bool = False) -> FastAPI:
    app.state.server = server
    port = port
    print(f"API server running on localhost:{port}")
    await serve_http(app, port, localmode=localmode)
