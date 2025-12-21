import asyncio
import json
import base64
import audioop
import uvicorn
import logging
import os
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from google import genai
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv # <--- ADDED THIS

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file or environment variables")

PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

# --- LOGGING ---
# Reduced logging level to WARNING to save I/O cycles during streaming
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GeminiLive")
logger.setLevel(logging.INFO)

app = FastAPI()

client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# --- AUDIO CONVERSION ---
def twilio_to_gemini(media_payload):
    try:
        mulaw_data = base64.b64decode(media_payload)
        pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
        return audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]
    except Exception:
        return None

def gemini_to_twilio(pcm_data):
    try:
        pcm_8k = audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]
        return audioop.lin2ulaw(pcm_8k, 2)
    except Exception:
        return None

# --- WEBHOOK ---
@app.post("/twiml")
async def twiml_response(request: Request):
    host = request.headers.get("host")
    return HTMLResponse(content=f"""
    <Response>
        <Connect>
            <Stream url="wss://{host}/media-stream" />
        </Connect>
        <Pause length="3600" />
    </Response>
    """, media_type="application/xml")

# --- WEBSOCKET HANDLER ---
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("📞 Twilio Connected")

    # OPTIMIZATION: Limit queue size to drop old packets if network lags
    audio_input_queue = asyncio.Queue(maxsize=5)
    stream_sid = None
    
    # OPTIMIZATION: Lock ensures 'clear' doesn't collide with 'media'
    socket_lock = asyncio.Lock()

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
        }
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info(f"🤖 Connected to {MODEL}")

            # 1. SEND GREETING
            await session.send(input="Hello! Introduce yourself.", end_of_turn=True)
            
            # --- RECEIVE FROM TWILIO ---
            async def receive_from_twilio():
                nonlocal stream_sid
                try:
                    while True:
                        try:
                            message = await websocket.receive_text()
                        except Exception:
                            break # Connection closed

                        data = json.loads(message)

                        if data["event"] == "media":
                            # Optimization: Decode directly, no prints
                            pcm_16k = twilio_to_gemini(data["media"]["payload"])
                            if pcm_16k:
                                # Non-blocking put: if full, drop oldest packet (Anti-Lag)
                                if audio_input_queue.full():
                                    try: audio_input_queue.get_nowait()
                                    except: pass
                                await audio_input_queue.put(pcm_16k)
                        
                        elif data["event"] == "start":
                            stream_sid = data["start"]["streamSid"]
                        elif data["event"] == "stop":
                            break
                except Exception as e:
                    logger.error(f"Twilio Rx Error: {e}")
                finally:
                    await audio_input_queue.put(None)

            # --- SEND TO GEMINI ---
            async def send_to_gemini():
                try:
                    while True:
                        pcm_data = await audio_input_queue.get()
                        if pcm_data is None: break
                        
                        await session.send_realtime_input(
                            audio={"data": pcm_data, "mime_type": "audio/pcm"}
                        )
                except Exception as e:
                    logger.error(f"Gemini Tx Error: {e}")

            # --- RECEIVE FROM GEMINI ---
            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content is None:
                                continue

                            # 1. Handle Audio
                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    if part.inline_data:
                                        # Optimization: Removed print("🔊")
                                        pcm_24k = part.inline_data.data
                                        mulaw_data = gemini_to_twilio(pcm_24k)
                                        
                                        if stream_sid and mulaw_data:
                                            async with socket_lock:
                                                await websocket.send_json({
                                                    "event": "media",
                                                    "streamSid": stream_sid,
                                                    "media": {
                                                        "payload": base64.b64encode(mulaw_data).decode("utf-8")
                                                    }
                                                })
                            
                            # 2. Handle Interruptions
                            if response.server_content.interrupted:
                                logger.info("🛑 Interrupted") # Keep this one log
                                if stream_sid:
                                    # Optimization: Safe clear
                                    try:
                                        async with socket_lock:
                                            await websocket.send_json({
                                                "event": "clear", 
                                                "streamSid": stream_sid
                                            })
                                    except Exception:
                                        pass 

                except Exception as e:
                    logger.error(f"Gemini Rx Error: {e}")

            # Run all tasks
            await asyncio.gather(receive_from_twilio(), send_to_gemini(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Connection Error: {e}")
    finally:
        logger.info("\nCall Ended")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)