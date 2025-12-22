import uvicorn
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("\n🔍 INCOMING CONNECTION REQUEST...")
    await websocket.accept()
    print("✅ EXOTEL CONNECTED SUCCESSFUL!")
    
    try:
        while True:
            # Just listen and print confirmation that data is flowing
            data = await websocket.receive_text()
            print("📩 Received Packet from Exotel")
    except Exception as e:
        print(f"❌ Connection Closed: {e}")

if __name__ == "__main__":
    print("🚀 Debug Server Running on Port 5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)