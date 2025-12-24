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

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Using Flash model for low latency
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Check your .env file.")

# Port configuration for Render
PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

# --- EXOTEL AUDIO SPECS ---
# Exotel expects 8kHz 16-bit PCM (mulaw or raw).
# We chunk response audio to ensure smooth playback on telephony.
CHUNK_SIZE = 3200  # 3200 bytes is approx 200ms of audio at 16khz/16bit or 8khz depending on format
PACING_INTERVAL = 0.185 # Slight delay between chunks to prevent jitter
SILENCE_THRESHOLD = 800 # RMS Amplitude threshold for logging "User is speaking"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ExotelGemini")

app = FastAPI()
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# --- AUDIO UTILITIES ---

def telephony_to_gemini(media_payload):
    """
    Decodes Exotel audio (Base64 -> PCM) and upsamples to 16kHz for Gemini.
    """
    try:
        # Exotel usually sends G.711 u-law or raw PCM at 8kHz.
        # Assuming raw PCM based on previous context.
        pcm_8k = base64.b64decode(media_payload)
        
        # Rate conversion: 8000Hz -> 16000Hz
        # (pcm_data, width, channels, input_rate, output_rate, state)
        return audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]
    except Exception as e:
        logger.error(f"Audio Decode Error: {e}")
        return None

def gemini_to_telephony(pcm_data):
    """
    Downsamples Gemini audio (24kHz PCM) to 8kHz for Exotel.
    """
    try:
        # Rate conversion: 24000Hz -> 8000Hz
        return audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]
    except Exception as e:
        logger.error(f"Audio Encode Error: {e}")
        return None

@app.get("/")
async def health_check():
    return {"status": "active", "service": "Exotel AI Server"}

# --- MAIN WEBSOCKET HANDLER ---

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    
    # --- 1. ROBUST CONTEXT EXTRACTION ---
    # We try to get parameters from the URL. 
    query_params = dict(websocket.query_params)
    
    # Log connection details for debugging
    logger.info(f"📞 New Connection. Params: {query_params}")

    # Extract or Default
    user_name = query_params.get("name", "Valued Customer")
    call_notes = query_params.get("notes", "General inquiry")
    
    # Sanitize inputs (Handle double encoding like 'John%20Doe' -> 'John Doe')
    user_name = user_name.replace("+", " ").replace("%20", " ")
    call_notes = call_notes.replace("+", " ").replace("%20", " ")

    logger.info(f"✅ AI Context Ready -> Name: {user_name} | Goal: {call_notes}")

    # Shared State
    audio_input_queue = asyncio.Queue(maxsize=50) # Increased buffer size
    stream_sid = None
    stream_key_name = "stream_sid" 
    is_speaking = False 
    socket_lock = asyncio.Lock()

    # --- 2. SYSTEM PROMPT ---
    SYSTEM_INSTRUCTION = f"""
    You are an AI sales assistant for AdRolls. You are speaking on the phone with {user_name}.
    
    CALL OBJECTIVE: {call_notes}.
    
    INSTRUCTIONS:
    1. Start by greeting {user_name} and immediately mentioning the reason for the call ({call_notes}).
    2. Keep your responses SHORT and CONCISE (1-2 sentences maximum). This is a voice call.
    3. Be professional, friendly, and helpful.
    4. If the user is silent or just says "hello", re-state the purpose of the call.
    """

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}} 
        },
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]}
    }

    try:
        # Connect to Gemini Live
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info("🔹 Connected to Google Gemini Live")
            
            # --- INITIAL GREETING TRIGGER ---
            # We send a text input to Gemini to force it to speak first with the context.
            initial_prompt = f"Hello {user_name}, I am calling from AdRolls regarding {call_notes}."
            await session.send(input=initial_prompt, end_of_turn=True)

            # --- TASK 1: RECEIVE AUDIO FROM EXOTEL ---
            async def receive_from_exotel():
                nonlocal stream_sid, stream_key_name
                last_speech_log = 0
                
                try:
                    while True:
                        # Receive message from Exotel
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        
                        event_type = data.get("event")

                        if event_type == "start":
                            # Exotel uses different keys sometimes (stream_sid vs streamSid)
                            stream_key_name = "stream_sid" if "stream_sid" in data else "streamSid"
                            stream_sid = data.get(stream_key_name)
                            if not stream_sid:
                                stream_sid = data.get("start", {}).get("stream_sid")
                            
                            logger.info(f"🔑 Stream Started. SID: {stream_sid}")

                        elif event_type == "media":
                            # Extract audio payload
                            payload = data["media"]["payload"]
                            # Convert to 16kHz PCM
                            pcm_16k = telephony_to_gemini(payload)
                            
                            if pcm_16k:
                                # --- LATENCY FIX: NO SILENCE BLOCKING ---
                                # We pass ALL audio to Gemini so it hears background noise/breaths.
                                # But we only LOG if it's loud, to keep terminal clean.
                                rms = audioop.rms(pcm_16k, 2)
                                
                                current_time = time.time()
                                if rms > SILENCE_THRESHOLD:
                                    if (current_time - last_speech_log) > 3:
                                        logger.info(f"🎤 Voice Detected (RMS: {rms})")
                                        last_speech_log = current_time

                                # Add to queue for the Sender Task
                                if audio_input_queue.full():
                                    try:
                                        audio_input_queue.get_nowait() # Drop oldest if full
                                    except:
                                        pass
                                await audio_input_queue.put(pcm_16k)
                        
                        elif event_type == "stop":
                            logger.info("🛑 Exotel sent Stop Event")
                            break
                            
                except WebSocketDisconnect:
                    logger.info("⚠️ Exotel WebSocket Disconnected")
                    raise # Propagate to main try/except
                except Exception as e:
                    logger.error(f"Error receiving from Exotel: {e}")
                finally:
                    # Signal other tasks to stop
                    await audio_input_queue.put(None)

            # --- TASK 2: SEND AUDIO TO GEMINI ---
            async def send_to_gemini():
                try:
                    while True:
                        # Get audio chunk from queue
                        pcm_data = await audio_input_queue.get()
                        if pcm_data is None: 
                            break # Stop signal received
                        
                        # Send to Gemini
                        await session.send_realtime_input(
                            audio={"data": pcm_data, "mime_type": "audio/pcm"}
                        )
                except Exception as e:
                    logger.error(f"Error sending to Gemini: {e}")

            # --- TASK 3: RECEIVE AUDIO FROM GEMINI & PLAYBACK ---
            async def receive_from_gemini():
                nonlocal is_speaking
                out_buffer = b""
                
                try:
                    while True:
                        async for response in session.receive():
                            # 1. Handle Interruptions (Barge-In)
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ User Interrupted AI. Clearing Buffer.")
                                is_speaking = False
                                out_buffer = b"" 
                                
                                if stream_sid:
                                    # Tell Exotel to clear its internal buffer immediately
                                    async with socket_lock:
                                        try:
                                            await websocket.send_json({
                                                "event": "clear",
                                                stream_key_name: stream_sid
                                            })
                                        except Exception:
                                            pass # Socket might be closed
                                continue

                            # 2. Handle Audio Response
                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        if not is_speaking:
                                            logger.info("🗣️ AI Speaking...")
                                            is_speaking = True
                                        
                                        # Convert 24k -> 8k
                                        pcm_24k = part.inline_data.data
                                        chunk_8k = gemini_to_telephony(pcm_24k)
                                        
                                        if chunk_8k:
                                            out_buffer += chunk_8k

                                        # Buffer and Chunk logic for smooth playback
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
                                                            "media": {
                                                                "payload": payload_str,
                                                                "track": "outbound"
                                                            }
                                                        })
                                                    except Exception:
                                                        break # Stop if socket failed
                                                
                                                # Pacing is critical for telephony jitter buffers
                                                await asyncio.sleep(PACING_INTERVAL)
                            
                            # 3. Detect End of Turn
                            if server_content and server_content.turn_complete:
                                is_speaking = False
                                # Flush remaining buffer if necessary? 
                                # Usually better to keep it for next chunk to avoid small audio artifacts.

                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            # --- RUN TASKS CONCURRENTLY ---
            await asyncio.gather(
                receive_from_exotel(),
                send_to_gemini(),
                receive_from_gemini()
            )

    except WebSocketDisconnect:
        logger.info("👋 WebSocket Disconnected cleanly.")
    except Exception as e:
        logger.error(f"🔥 Critical Exception: {e}")
    finally:
        logger.info("🔒 Closing resources...")
        # Use a safe close pattern to avoid "Unexpected ASGI message" errors
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except Exception:
            pass
        logger.info("✅ Session Closed.")

if __name__ == '__main__':
    # Render provides PORT. Localhost defaults to 5000.
    uvicorn.run(app, host=HOST, port=PORT)