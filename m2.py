# ================== MULTILINGUAL VOICE RAG BACKEND (ERROR-FREE) ==================
# FastAPI + Deepgram STT + Groq LLM + Pinecone RAG + ElevenLabs TTS
# Python 3.9+

import os
import base64
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import httpx

# ----------------- OPTIONAL CLIENTS -----------------
try:
    from groq import AsyncGroq
except Exception:
    AsyncGroq = None

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from elevenlabs.client import AsyncElevenLabs
except Exception:
    AsyncElevenLabs = None

# ----------------- CONFIG -----------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") # mind  
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") #knowledge base cat =[0.1,0.2,0.7]
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") # text to speech
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") # speech to text
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT") 

if not all([GROQ_API_KEY, PINECONE_API_KEY, ELEVENLABS_API_KEY, DEEPGRAM_API_KEY, PINECONE_ENV]):
    raise RuntimeError("Missing one or more required environment variables")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") #
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")# input > hii --vector embedding --> pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-hf-index")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))
PINECONE_TOP_K = int(os.getenv("PINECONE_TOP_K", "3"))

DEFAULT_STT_LANGUAGE = os.getenv("DEFAULT_STT_LANGUAGE", "en-IN")

ELEVEN_LANGS = {
    "ar","bg","cmn","zh","hr","cs","da","nl","en","fil","fi","fr",
    "de","el","hi","id","ja","it","ko","ms","pl","pt","ro","sk",
    "es","sv","ta","tr","uk","ru"
}

DEEPGRAM_NOVA3_LANGS = {
    "en","es","fr","de","hi","ru","pt","ja","it","nl",
    "bg","cs","fi","hu","ko","pl","uk","vi"
}

LANG_LABELS = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "pl": "Polish", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "cmn": "Chinese", "ar": "Arabic", "tr": "Turkish",
    "bg": "Bulgarian", "ro": "Romanian", "cs": "Czech", "sk": "Slovak",
    "fi": "Finnish", "el": "Greek", "id": "Indonesian", "ms": "Malay",
    "uk": "Ukrainian", "fil": "Filipino", "vi": "Vietnamese",
}


def lang_label(code: str) -> str:
    return LANG_LABELS.get(code, "English")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multilingual_voice_rag")

# ----------------- REQUEST SCHEMA -----------------
class TextChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    language: str = "en"
    include_audio: bool = True

# ----------------- LANGUAGE HELPERS -----------------
def normalize_lang_for_llm_tts(lang: str) -> str:
    if not lang:
        return "en"
    base = lang.lower().split("-")[0]
    return base if base in ELEVEN_LANGS else "en"


def normalize_language_for_deepgram(lang: str) -> str:
    if not lang:
        return "en"
    return lang.lower().split("-")[0]


def choose_deepgram_model_and_tier(code: str) -> Dict[str, Optional[str]]:
    if code in {"ta", "te"}:
        return {"model": "general", "tier": "enhanced"}
    if code in DEEPGRAM_NOVA3_LANGS:
        return {"model": "nova-3", "tier": None}
    return {"model": "general", "tier": None}

# ----------------- CORE SERVICES -----------------
class Services:
    def __init__(self):
        self.client_groq = AsyncGroq(api_key=GROQ_API_KEY) if AsyncGroq else None
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL) if SentenceTransformer else None

        if Pinecone:
            pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            self.index = pc.Index(PINECONE_INDEX_NAME)
        else:
            self.index = None

        self.tts = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY) if AsyncElevenLabs else None
        self.history: Dict[str, List[Dict[str, str]]] = {}
        self.lock = asyncio.Lock()

    # --------- EMBEDDINGS ---------
    def embed(self, text: str) -> List[float]:
        if not self.embed_model:
            vec = [0.0] * VECTOR_DIMENSION
            for i, c in enumerate(text[:VECTOR_DIMENSION]):
                vec[i] = (ord(c) % 10) / 10.0
            return vec
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()

    def store_conversation(self, user: str, assistant: str, session_id: str, language: str):
        if not self.index:
            return
        now = datetime.now().isoformat()
        text = f"User ({language}): {user}\nAssistant: {assistant}"
        self.index.upsert([
            (
                f"conv-{session_id}-{now}",
                self.embed(text),
                {
                    "content": text,
                    "session_id": session_id,
                    "language": language,
                    "timestamp": now,
                },
            )
        ])

    async def get_context(self, query: str, session_id: str) -> str:
        if not self.index:
            return ""
        res = self.index.query(
            vector=self.embed(query),
            top_k=PINECONE_TOP_K,
            include_metadata=True,
            filter={"session_id": session_id},
        )
        parts = []
        for m in res.get("matches", []):
            md = m.get("metadata", {})
            if md.get("content"):
                parts.append(md["content"])
        return "\n".join(parts)

    # --------- HISTORY ---------
    async def add_history(self, session_id: str, role: str, content: str):
        async with self.lock:
            self.history.setdefault(session_id, []).append({"role": role, "content": content})
            self.history[session_id] = self.history[session_id][-8:]

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        async with self.lock:
            return list(self.history.get(session_id, []))

    # --------- LLM ---------
    async def generate_answer(self, msg: str, ctx: str, session_id: str, language: str) -> str:
        messages = [{
            "role": "system",
            "content": f"You are a helpful assistant. Answer briefly in {lang_label(language)}.",
        }]
        if ctx:
            messages.append({"role": "system", "content": f"CONTEXT:\n{ctx}"})
        messages.extend(await self.get_history(session_id))
        messages.append({"role": "user", "content": msg})

        if not self.client_groq:
            answer = msg
        else:
            resp = await self.client_groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
            )
            answer = resp.choices[0].message.content

        await self.add_history(session_id, "user", msg)
        await self.add_history(session_id, "assistant", answer)
        return answer

    # --------- TTS ---------
    async def synth(self, text: str, language: str) -> Optional[str]:
        if not self.tts:
            return None
        audio = b""
        async for c in self.tts.text_to_speech.convert(
            text=text,
            voice_id=os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            language_code=language,
        ):
            audio += c
        return base64.b64encode(audio).decode() if audio else None

services = Services()

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Multilingual Voice RAG Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- HEALTH -----------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ----------------- TEXT CHAT -----------------
@app.post("/api/chat/text")
async def chat_text(req: TextChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    lang = normalize_lang_for_llm_tts(req.language)
    ctx = await services.get_context(req.message, req.session_id)
    ans = await services.generate_answer(req.message, ctx, req.session_id, lang)
    services.store_conversation(req.message, ans, req.session_id, lang)

    res = {
        "success": True,
        "answer": ans,
        "language": lang,
        "session_id": req.session_id,
        "timestamp": datetime.now().isoformat(),
    }

    if req.include_audio:
        audio = await services.synth(ans, lang)
        if audio:
            res["audio_base64"] = audio
            res["audio_format"] = "mp3"

    return JSONResponse(res)

# ----------------- AUDIO CHAT -----------------
@app.post("/api/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
    language_hint: str = Form(DEFAULT_STT_LANGUAGE),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio")

    dg_lang = normalize_language_for_deepgram(language_hint)
    model_info = choose_deepgram_model_and_tier(dg_lang)
    params={"language": dg_lang, 
            "model": model_info["model"],
            "puntuate": "true",
            "smart_format":"true",

    }
    if model_info["tier"] is not None:
        params["tier"]=model_info["tier"]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/listen",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}","content-type": "audio/webm"},
            params=params,
            content=audio_bytes,
        )

    if r.status_code != 200:
        raise HTTPException(500, r.text)

    data = r.json()
    transcript = (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )

    if not transcript:
        raise HTTPException(400, "No speech detected")

    lang = normalize_lang_for_llm_tts(language_hint)
    ctx = await services.get_context(transcript, session_id)
    ans = await services.generate_answer(transcript, ctx, session_id, lang)
    services.store_conversation(transcript, ans, session_id, lang)

    audio_out = await services.synth(ans, lang)

    return JSONResponse({
        "success": True,
        "transcript": transcript,
        "answer": ans,
        "language": lang,
        "audio_base64": audio_out,
        "audio_format": "mp3",
    })

# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
