import asyncio
import json
import base64
import audioop
import uvicorn
import logging
import os
import time
import sys
from fastapi import FastAPI, WebSocket, Request
from fastapi.websockets import WebSocketDisconnect
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL = "gemini-2.0-flash-exp"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

# Audio Settings
CHUNK_SIZE = 3200 
PACING_INTERVAL = 0.185
# Increased threshold to ignore background noise/breathing
SILENCE_THRESHOLD = 1000 

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ExotelGemini")

app = FastAPI()
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# --- HELPER FUNCTIONS ---
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

@app.get("/")
async def health_check():
    return {"status": "active", "service": "AdRolls AI Voice Agent"}

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    
    # --- 1. ROBUST CONTEXT EXTRACTION ---
    # Log everything to debug why name might be missing
    query_params = dict(websocket.query_params)
    logger.info(f"🔍 Connection Query Params: {query_params}")
    
    # Try getting name from Query Params, default to "Valued Customer"
    user_name = query_params.get("name", "Valued Customer")
    call_notes = query_params.get("notes", "General inquiry")
    
    # Fix for double-encoded URLs (sometimes spaces become + or %20)
    user_name = user_name.replace("+", " ").replace("%20", " ")
    call_notes = call_notes.replace("+", " ").replace("%20", " ")

    logger.info(f"✅ AI Initialized for: {user_name} | Topic: {call_notes}")

    audio_input_queue = asyncio.Queue(maxsize=10)
    stream_sid = None
    stream_key_name = "stream_sid" 
    is_speaking = False 
    socket_lock = asyncio.Lock()

    # --- 2. SYSTEM INSTRUCTION ---
    SYSTEM_INSTRUCTION = f"""
    You are an AI sales assistant for AdRolls. You are currently on a phone call with {user_name}.
    
    CONTEXT & GOAL:
    The purpose of this call is: {call_notes}.
    
    RULES:
    1. START IMMEDIATELY: As soon as you hear the user, acknowledge the purpose of the call.
    2. BE CONCISE: Use short sentences (under 15 words). No long monologues.
    3. BE NATURAL: If the user says "Hello?", reply "Hi {user_name}, I'm calling from AdRolls about {call_notes}."
    """

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
        },
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]}
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info("🔹 Gemini Live Session Connected")
            
            # Send initial Greeting so AI knows to start
            await session.send(input=f"Hi, I am calling from AdRolls about {call_notes}. Is this {user_name}?", end_of_turn=True)

            # --- EXOTEL LISTENER TASK ---
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
                            payload = data["media"]["payload"]
                            pcm_16k = telephony_to_gemini(payload)
                            
                            if pcm_16k:
                                # SILENCE GATE: Only process if volume > threshold
                                rms = audioop.rms(pcm_16k, 2)
                                
                                # Log occasionally to prove it's working, but only if loud enough
                                if rms > SILENCE_THRESHOLD:
                                    current_time = time.time()
                                    if current_time - last_speech_log > 5:
                                        logger.info(f"🎤 Voice detected (RMS: {rms})")
                                        last_speech_log = current_time
                                    
                                    # Send to buffer
                                    if audio_input_queue.full():
                                        audio_input_queue.get_nowait()
                                    await audio_input_queue.put(pcm_16k)
                        
                        elif data.get("event") == "stop":
                            logger.info("🛑 Exotel sent Stop event")
                            break
                except WebSocketDisconnect:
                    logger.info("⚠️ WebSocket Disconnected by Client")
                    raise
                except Exception as e:
                    logger.error(f"Exotel Rx Error: {e}")
                finally:
                    await audio_input_queue.put(None)

            # --- GEMINI SENDER TASK ---
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

            # --- GEMINI RECEIVER TASK ---
            async def receive_from_gemini():
                nonlocal is_speaking
                out_buffer = b""
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ AI Interrupted")
                                is_speaking = False
                                out_buffer = b"" 
                                if stream_sid:
                                    async with socket_lock:
                                        # Safe send, ignore if socket closed
                                        try:
                                            await websocket.send_json({
                                                "event": "clear",
                                                stream_key_name: stream_sid
                                            })
                                        except: pass
                                continue

                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        if not is_speaking:
                                            logger.info("🗣️ AI Speaking response...")
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
                                                    try:
                                                        await websocket.send_json({
                                                            "event": "media",
                                                            stream_key_name: stream_sid,
                                                            "media": {"payload": payload_str}
                                                        })
                                                    except Exception:
                                                        break # Socket likely closed
                                                await asyncio.sleep(PACING_INTERVAL)
                            
                            if server_content and server_content.turn_complete:
                                is_speaking = False

                except Exception as e:
                    logger.error(f"Gemini Rx Error: {e}")

            # Run all tasks
            await asyncio.gather(receive_from_exotel(), send_to_gemini(), receive_from_gemini())

    except WebSocketDisconnect:
        logger.info("👋 Client Disconnected")
    except Exception as e:
        logger.error(f"🔥 Critical Server Error: {e}")
    finally:
        logger.info("🔒 Closing Connection")
        # SAFE CLOSE PATTERN to prevent RuntimeError
        try:
            await websocket.close()
        except RuntimeError:
            pass # Already closed
        except Exception as e:
            logger.info(f"Socket close ignored: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)