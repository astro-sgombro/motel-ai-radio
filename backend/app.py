"""
app.py — Custom Motel Radio Backend
-----------------------------------
Endpoint principali:
- POST /api/tts                   → genera MP3 via ElevenLabs (similarity_boost=0.34)
- POST /api/intro-from-track-text → dato un link Spotify (singola traccia) restituisce SOLO il testo intro
- POST /api/intro-from-track      → dato un link Spotify (singola traccia) restituisce MP3 dell'intro
- GET  /api/voices                → elenco voci ElevenLabs
- GET  /health                    → diagnostica rapida

Requisiti .env:
ELEVENLABS_API_KEY=...
DEFAULT_VOICE_ID=...
MODEL_ID=eleven_flash_v2_5
ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
"""

import os
import re
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# =========================
# Caricamento configurazione
# =========================
load_dotenv()

# --- ElevenLabs ---
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE_ID")           # ID voce di default
MODEL_ID = os.getenv("MODEL_ID", "eleven_flash_v2_5")   # default al modello Flash v2.5
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "mp3_44100_128")
SIMILARITY_BOOST = 0.34                                  # hardcoded richiesto

# --- CORS ---
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# --- Spotify (Client Credentials) ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# =========================
# Inizializzazione FastAPI
# =========================
app = FastAPI(title="Custom Motel Radio - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========
# UTIL SPOTIFY
# ==========

def extract_track_id(track_url: str) -> str:
    """
    Estrae l'ID della traccia Spotify da:
      - https://open.spotify.com/track/{id}
      - https://open.spotify.com/intl-xx/track/{id}?...
      - spotify:track:{id}
    Ritorna '' se non valido.
    """
    if not track_url:
        return ""

    track_url = track_url.strip()

    # Caso URI: spotify:track:{id}
    m = re.match(r"^spotify:track:([A-Za-z0-9]{22})$", track_url)
    if m:
        return m.group(1)

    # Caso URL: open.spotify.com con o senza 'intl-xx' e query
    try:
        u = urlparse(track_url)
    except Exception:
        return ""

    if "open.spotify.com" not in u.netloc:
        return ""

    parts = [p for p in u.path.split("/") if p]
    # Funziona per '/track/{id}' e '/intl-it/track/{id}'
    if len(parts) >= 2 and parts[-2] == "track":
        candidate = parts[-1]
    else:
        return ""

    return candidate if re.fullmatch(r"[A-Za-z0-9]{22}", candidate) else ""


# Sostituisci la tua funzione con questa
async def get_spotify_token() -> str:
    """
    Ottiene un access_token Spotify via Client Credentials.
    Ritorna 500 solo se mancano le credenziali; se Spotify risponde con errore,
    rilancia lo status code originale con un messaggio parlante.
    """
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        # Errore di configurazione BE → 500 chiaro
        raise HTTPException(
            status_code=500,
            detail="Spotify client credentials mancanti. Configura SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET nel .env."
        )

    token_url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    auth = (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(token_url, data=data, auth=auth)

    if r.status_code != 200:
        # Passa lo status code di Spotify e aggiungi il body per capire subito il motivo
        raise HTTPException(
            status_code=r.status_code,
            detail=f"Spotify token error: {r.text[:300]}"
        )

    return r.json()["access_token"]

# Sostituisci la tua funzione con questa
async def fetch_track_info(track_id: str, token: str) -> dict:
    """
    Recupera metadati brano e prova a prendere le audio-features.
    Se le features falliscono, NON esplode: usa valori di default neutri.
    """
    base = "https://api.spotify.com/v1"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=20) as client:
        tr = await client.get(f"{base}/tracks/{track_id}", headers=headers)
        if tr.status_code != 200:
            # Propaga lo status code (404, 401, ecc.) con messaggio leggibile
            raise HTTPException(tr.status_code, f"Spotify tracks error: {tr.text[:300]}")
        track = tr.json()

        af = await client.get(f"{base}/audio-features/{track_id}", headers=headers)
        features = {}
        if af.status_code == 200:
            features = af.json()
        else:
            # Fallback: valori neutri se endpoint features non disponibile/404
            features = {"energy": 0.5, "valence": 0.5, "danceability": 0.5, "tempo": 0}

    # Parsing robusto dei metadati (senza dare per scontate le chiavi)
    title = track.get("name") or "Unknown title"
    artists = track.get("artists") or []
    artist = (artists[0].get("name") if artists and isinstance(artists[0], dict) else "Unknown artist")
    album_obj = track.get("album") or {}
    album = album_obj.get("name") or "Unknown album"
    release_date = album_obj.get("release_date") or ""
    year = release_date[:4] if release_date else "—"

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "year": year,
        "energy": float(features.get("energy", 0.5)),
        "valence": float(features.get("valence", 0.5)),
        "danceability": float(features.get("danceability", 0.5)),
        "tempo": float(features.get("tempo", 0)),
    }



def build_intro_text(meta: dict) -> str:
    """
    Genera micro-testo 'Custom Motel Radio':
    - 1 fatto (anno + artista + album + titolo)
    - 1-2 parole di mood mappate da energy/valence/danceability
    Testo breve e scorrevole per TTS.
    """
    mood_bits = []
    if meta["energy"] >= 0.65:
        mood_bits.append("energia alta")
    elif meta["energy"] <= 0.35:
        mood_bits.append("atmosfere soffuse")

    if meta["valence"] >= 0.65:
        mood_bits.append("vibe luminosa")
    elif meta["valence"] <= 0.35:
        mood_bits.append("tinte più scure")

    if meta["danceability"] >= 0.65:
        mood_bits.append("groove che chiama movimento")

    mood = ", ".join(mood_bits) if mood_bits else "equilibrio perfetto"

    # Punteggiatura essenziale per resa naturale in TTS
    return (
        f"{meta['year']}, {meta['artist']} dal disco {meta['album']}: "
        f"{meta['title']}. {mood}. Buon ascolto."
    )

# ==================
# MODELLI DI RICHIESTA
# ==================

class TrackURL(BaseModel):
    """Schema della richiesta per le intro da link a SINGOLA traccia Spotify."""
    track_url: str


# ==========
# ENDPOINT TTS (MP3)
# ==========

@app.post("/api/tts")
async def tts(payload: dict = Body(...)):
    """
    Genera TTS (non-streaming) usando ElevenLabs.
    Accetta:
      - text (obbligatorio)
      - voiceId (opzionale; default dal .env)
      - language_code (opzionale; usato dai modelli Turbo/Flash v2.5; default 'en')
      - voice_settings (opzionale; verrà mergiato ma similarity_boost è forzato a 0.34)
    Ritorna: audio/mpeg (MP3).
    """
    text = (payload or {}).get("text", "").strip()
    voice_id = (payload or {}).get("voiceId", DEFAULT_VOICE)
    lang = (payload or {}).get("language_code", "en")
    incoming_voice_settings = (payload or {}).get("voice_settings", {})

    if not ELEVEN_KEY or not text:
        raise HTTPException(400, "Missing API key or text")
    if not voice_id:
        raise HTTPException(400, "Missing voiceId (and DEFAULT_VOICE_ID not set)")

    # Merge non distruttivo + override similarity_boost
    voice_settings = {
        **incoming_voice_settings,           # preserva altri parametri (es. stability) se passati
        "similarity_boost": SIMILARITY_BOOST # forza 34% come richiesto
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={OUTPUT_FORMAT}"
    headers = {
        "xi-api-key": ELEVEN_KEY,
        "accept": "audio/mpeg",              # richiedi MP3
        "content-type": "application/json",
    }
    body = {
        "text": text,
        "model_id": MODEL_ID,                # es. "eleven_flash_v2_5"
        "language_code": lang,               # supportato dai modelli Turbo/Flash v2.5
        "voice_settings": voice_settings
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"ElevenLabs error: {r.text[:500]}")

    return Response(content=r.content, media_type="audio/mpeg")


# ===========================
# ENDPOINT SOLO TESTO (JSON)
# ===========================

# Mantieni il tuo modello Pydantic:
# class TrackURL(BaseModel):
#     track_url: str

@app.post("/api/intro-from-track-text")
async def intro_from_track_text(req: TrackURL):
    """
    Dato un link a una SINGOLA traccia Spotify, genera e restituisce SOLO il testo dell'intro (JSON).
    Tutti gli errori prevedibili escono come HTTPException con messaggio esplicito.
    """
    track_url = (req.track_url or "").strip()
    if not track_url:
        raise HTTPException(400, "Parametro 'track_url' mancante o vuoto.")

    track_id = extract_track_id(track_url)
    if not track_id:
        raise HTTPException(400, "URL non valido o non relativo a una singola traccia Spotify.")

    try:
        token = await get_spotify_token()
        meta = await fetch_track_info(track_id, token)
    except HTTPException as e:
        # Rilancia status e messaggio originali (già parlanti)
        raise e
    except Exception as e:
        # Qualsiasi altro errore imprevisto → 502 con dettaglio
        raise HTTPException(502, f"Errore imprevisto durante il recupero dati Spotify: {str(e)}")

    text = build_intro_text(meta)

    return JSONResponse({
        "ok": True,
        "track_id": track_id,
        "text": text,
        "meta": meta
    })



# =============================================
# ENDPOINT INTRO COMPLETA (TESTO → MP3 ElevenLabs)
# =============================================

@app.post("/api/intro-from-track")
async def intro_from_track(req: TrackURL):
    """
    Dato un link a una SINGOLA traccia Spotify, genera il testo intro e
    restituisce un MP3 tramite ElevenLabs (uso della voce di default).
    """
    track_url = (req.track_url or "").strip()
    if not track_url:
        raise HTTPException(400, "Parametro 'track_url' mancante o vuoto.")

    track_id = extract_track_id(track_url)
    if not track_id:
        raise HTTPException(400, "URL non valido o non relativo a una singola traccia Spotify.")

    if not (ELEVEN_KEY and DEFAULT_VOICE):
        raise HTTPException(500, "Config ElevenLabs mancante (API key o default voice).")

    token = await get_spotify_token()
    meta = await fetch_track_info(track_id, token)
    text = build_intro_text(meta)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE}?output_format={OUTPUT_FORMAT}"
    headers = {
        "xi-api-key": ELEVEN_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    eleven_body = {
        "text": text,
        "model_id": MODEL_ID,
        "language_code": "it",  # Intro in italiano per coerenza col template
        "voice_settings": {
            "similarity_boost": SIMILARITY_BOOST
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=eleven_body)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"ElevenLabs error: {r.text[:400]}")

    return Response(content=r.content, media_type="audio/mpeg")


# ==========
# VOCI & HEALTH
# ==========

@app.get("/api/voices")
async def voices():
    """Proxy semplice per elencare le voci disponibili su ElevenLabs."""
    if not ELEVEN_KEY:
        raise HTTPException(500, "Missing ELEVENLABS_API_KEY")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVEN_KEY}
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(url, headers=headers)
    if res.status_code != 200:
        raise HTTPException(res.status_code, res.text)
    return res.json()


@app.get("/health")
def health():
    return {
        "env_loaded": bool(ELEVEN_KEY),
        "has_default_voice": bool(DEFAULT_VOICE),
        "model_id": MODEL_ID,
        "output_format": OUTPUT_FORMAT,
        "similarity_boost": SIMILARITY_BOOST,
        "allowed_origins": ALLOWED,
        "spotify_client_set": bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET),
        "spotify_client_id_len": len(SPOTIFY_CLIENT_ID or ""),
        "spotify_client_secret_len": len(SPOTIFY_CLIENT_SECRET or ""),
    }
