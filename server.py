import asyncio
import json
import base64
import audioop
import uvicorn
import logging
import os
import time
import sys
from fastapi import FastAPI, WebSocket
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Using 2.0 Flash for significantly lower latency in real-time loops
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

# --- EXOTEL SPECS ---
# Exotel requires 16-bit PCM @ 8kHz. Minimum chunk size 3200 bytes (200ms).
CHUNK_SIZE = 3200 
# Pacing: Deliver audio slightly faster than real-time to prevent robotic sound
PACING_INTERVAL = 0.185

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ExotelGemini")

app = FastAPI()
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# --- AUDIO UTILS ---
def telephony_to_gemini(media_payload):
    try:
        # Exotel (16-bit PCM 8k) -> Gemini (PCM 16k)
        pcm_8k = base64.b64decode(media_payload)
        return audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]
    except Exception as e:
        logger.error(f"Decode Error: {e}")
        return None

def gemini_to_telephony(pcm_data):
    try:
        # Gemini (PCM 24k) -> Exotel (PCM 8k)
        return audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]
    except Exception as e:
        logger.error(f"Encode Error: {e}")
        return None

# --- WEBSOCKET HANDLER ---
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Exotel WebSocket Connected")

    audio_input_queue = asyncio.Queue(maxsize=10)
    stream_sid = None
    stream_key_name = "stream_sid" 
    is_speaking = False 
    socket_lock = asyncio.Lock()

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
        }
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info(f"🔹 Connected to Gemini: {MODEL}")
            
            # Initial AI greeting
            await session.send(input="Hello! How can I help you today?", end_of_turn=True)

            # --- 1. RECEIVE FROM EXOTEL (Handles incoming User Voice) ---
            async def receive_from_exotel():
                nonlocal stream_sid, stream_key_name
                last_speech_log = 0
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        
                        if data.get("event") == "start":
                            logger.info(f"📩 START EVENT: {json.dumps(data)}")
                            stream_key_name = "stream_sid" if "stream_sid" in data else "streamSid"
                            stream_sid = data.get("stream_sid") or data.get("streamSid") or data.get("start", {}).get("stream_sid")
                            logger.info(f"🔑 Stream ID Captured: {stream_sid}")

                        elif data.get("event") == "media":
                            # LOG USER SPEECH: Track if user audio is reaching the server
                            current_time = time.time()
                            if current_time - last_speech_log > 2:
                                logger.info("🎤 User is speaking... (Audio reaching server)")
                                last_speech_log = current_time

                            payload = data["media"]["payload"]
                            pcm_16k = telephony_to_gemini(payload)
                            if pcm_16k:
                                if audio_input_queue.full():
                                    audio_input_queue.get_nowait()
                                await audio_input_queue.put(pcm_16k)
                        
                        elif data.get("event") == "stop":
                            logger.info("🛑 Exotel Stop Signal Received")
                            break
                except Exception as e:
                    logger.error(f"Exotel Rx Error: {e}")
                finally:
                    await audio_input_queue.put(None)

            # --- 2. SEND TO GEMINI (Feed User Audio) ---
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

            # --- 3. RECEIVE FROM GEMINI (Handle AI Voice & Barge-in) ---
            async def receive_from_gemini():
                nonlocal is_speaking
                out_buffer = b""
                
                try:
                    while True:
                        async for response in session.receive():
                            # --- BARGE-IN DETECTION ---
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ BARGE-IN: User interrupted AI. Clearing buffers.")
                                is_speaking = False
                                out_buffer = b"" 
                                
                                # Immediately tell Exotel to clear its playback buffer
                                if stream_sid:
                                    async with socket_lock:
                                        await websocket.send_json({
                                            "event": "clear",
                                            stream_key_name: stream_sid
                                        })
                                continue

                            # --- AUDIO PROCESSING ---
                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        if not is_speaking:
                                            logger.info("🗣️ Gemini generating response...")
                                        is_speaking = True
                                        
                                        pcm_24k = part.inline_data.data
                                        chunk_8k = gemini_to_telephony(pcm_24k)
                                        if chunk_8k:
                                            out_buffer += chunk_8k

                                        # Strict 3200-byte chunking for Exotel stability
                                        while len(out_buffer) >= CHUNK_SIZE:
                                            to_send = out_buffer[:CHUNK_SIZE]
                                            out_buffer = out_buffer[CHUNK_SIZE:]
                                            
                                            if stream_sid:
                                                payload_str = base64.b64encode(to_send).decode("utf-8")
                                                async with socket_lock:
                                                    await websocket.send_json({
                                                        "event": "media",
                                                        stream_key_name: stream_sid,
                                                        "media": {"payload": payload_str}
                                                    })
                                            # Pace the packets
                                            await asyncio.sleep(PACING_INTERVAL)
                            
                            if server_content and server_content.turn_complete:
                                logger.info("🔹 Gemini turn complete.")
                                is_speaking = False

                except Exception as e:
                    logger.error(f"❌ Gemini Rx Error: {e}")

            # Run concurrency tasks
            await asyncio.gather(receive_from_exotel(), send_to_gemini(), receive_from_gemini())

    except Exception as e:
        logger.error(f"🔥 Critical Failure: {e}")
    finally:
        logger.info("👋 Call Session Closed")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)