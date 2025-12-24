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
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

# --- EXOTEL SPECS ---
CHUNK_SIZE = 3200 
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
        pcm_8k = base64.b64decode(media_payload)
        return audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]
    except Exception as e:
        logger.error(f"Decode Error: {e}")
        return None

def gemini_to_telephony(pcm_data):
    try:
        return audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]
    except Exception as e:
        logger.error(f"Encode Error: {e}")
        return None

# --- WEBSOCKET HANDLER ---
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Capture Context
    query_params = websocket.query_params
    user_name = query_params.get("name", "Valued Customer")
    call_notes = query_params.get("notes", "General inquiry")
    
    logger.info(f"✅ Connection: {user_name} | Goal: {call_notes}")

    audio_input_queue = asyncio.Queue(maxsize=10)
    stream_sid = None
    stream_key_name = "stream_sid" 
    is_speaking = False 
    socket_lock = asyncio.Lock()

    # 2. Stricter System Instruction
    SYSTEM_INSTRUCTION = f"""
    You are an AI sales assistant for AdRolls calling {user_name}.
    
    CRITICAL INSTRUCTION:
    The purpose of this call is: {call_notes}.
    
    You MUST mention this purpose immediately after verifying the user's name.
    Do not ask generic "How are you?" questions. Get straight to the point.
    
    Keep responses short (1-2 sentences).
    """

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
        },
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]}
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info(f"🔹 Connected to Gemini")
            
            # 3. Direct Initial Trigger
            await session.send(input=f"Hello, I am calling from AdRolls regarding {call_notes}. Am I speaking with {user_name}?", end_of_turn=True)

            # --- RECEIVE FROM EXOTEL ---
            async def receive_from_exotel():
                nonlocal stream_sid, stream_key_name
                last_speech_log = 0
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        
                        if data.get("event") == "start":
                            stream_key_name = "stream_sid" if "stream_sid" in data else "streamSid"
                            stream_sid = data.get("stream_sid") or data.get("streamSid") or data.get("start", {}).get("stream_sid")
                            logger.info(f"🔑 Stream Started: {stream_sid}")

                        elif data.get("event") == "media":
                            current_time = time.time()
                            if current_time - last_speech_log > 2:
                                logger.info("🎤 Receiving audio...")
                                last_speech_log = current_time

                            payload = data["media"]["payload"]
                            pcm_16k = telephony_to_gemini(payload)
                            if pcm_16k:
                                if audio_input_queue.full():
                                    audio_input_queue.get_nowait()
                                await audio_input_queue.put(pcm_16k)
                        
                        elif data.get("event") == "stop":
                            logger.info("🛑 Call Stopped")
                            break
                except Exception as e:
                    logger.error(f"Exotel Rx Error: {e}")
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
                nonlocal is_speaking
                out_buffer = b""
                
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ Interrupted")
                                is_speaking = False
                                out_buffer = b"" 
                                if stream_sid:
                                    async with socket_lock:
                                        await websocket.send_json({
                                            "event": "clear",
                                            stream_key_name: stream_sid
                                        })
                                continue

                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        if not is_speaking:
                                            logger.info("🗣️ AI Speaking")
                                            is_speaking = True
                                        
                                        pcm_24k = part.inline_data.data
                                        chunk_8k = gemini_to_telephony(pcm_24k)
                                        if chunk_8k:
                                            out_buffer += chunk_8k

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
                                                await asyncio.sleep(PACING_INTERVAL)
                            
                            if server_content and server_content.turn_complete:
                                is_speaking = False

                except Exception as e:
                    logger.error(f"Gemini Rx Error: {e}")

            await asyncio.gather(receive_from_exotel(), send_to_gemini(), receive_from_gemini())

    except Exception as e:
        logger.error(f"🔥 Critical Failure: {e}")
    finally:
        logger.info("👋 Session Ended")
        await websocket.close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)