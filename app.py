from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI()


# Serve React app (build)
client_path = Path("client/out")
app.mount("/static", StaticFiles(directory=client_path), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:

        while True:

            data = await websocket.receive_json()
            message=data["text"]
            print(f"Received message: {message}")

            await websocket.send_json([{"role":"ai", "content":"lorem ipsum blablabla"}])

    except WebSocketDisconnect:
        print("errore")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    target_path = client_path / full_path
    if target_path.is_dir():
        index = target_path / "index.html"
        if index.exists():
            return FileResponse(index)
    elif target_path.exists():
        return FileResponse(target_path)

    fallback_index = client_path / "index.html"
    return FileResponse(fallback_index)
