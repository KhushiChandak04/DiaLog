# --- Firestore Backend Logging ---
try:
    from firebase_admin_setup import db as firestore_db, firebase_initialized
    from firebase_admin import firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    print("⚠️ Firebase not available - running without cloud logging")
    firestore_db = None
    firebase_initialized = False
    FIREBASE_AVAILABLE = False

from pydantic import BaseModel
from improved_model_system import MealSafetyPredictor, RiskLevel, run_acceptance_tests

# Import personalized ML model
try:
    from personalized_ml_model import PersonalizedMealRecommender
    PERSONALIZED_ML_AVAILABLE = True
    print("✅ Personalized ML model imported successfully")
except ImportError as e:
    print(f"⚠️ Personalized ML model not available: {e}")
    PersonalizedMealRecommender = None
    PERSONALIZED_ML_AVAILABLE = False


# Pydantic model for meal log

# Accepts a list of meals per log
class MealLogMeal(BaseModel):
    meal_name: str
    quantity: int
    unit: str
    time_of_day: str

class MealLog(BaseModel):
    userId: str
    sugar_level_fasting: float
    sugar_level_post: float
    meals: list[MealLogMeal]
    createdAt: str = None  # Optional, can be set by backend




# backend/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
import hashlib

# Load environment variables from this backend folder regardless of CWD
BASE_DIR = Path(__file__).resolve().parent
ENV_FILES_LOADED = []
if (BASE_DIR / '.env').exists():
    load_dotenv(BASE_DIR / '.env', override=True)
    ENV_FILES_LOADED.append(str((BASE_DIR / '.env').resolve()))
elif (BASE_DIR / '.env.local').exists():
    load_dotenv(BASE_DIR / '.env.local', override=True)
    ENV_FILES_LOADED.append(str((BASE_DIR / '.env.local').resolve()))
else:
    load_dotenv(override=True)
    ENV_FILES_LOADED.append('process env only')

# Read and normalize Gemini settings
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # optional override, e.g., gemini-1.5-flash-latest
GEMINI_DEBUG_ERRORS = (os.getenv("GEMINI_DEBUG_ERRORS") or "").strip() == "1"
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except Exception as _gem_e:
    print("⚠️ Gemini not available:", _gem_e)
    genai = None
    GEMINI_AVAILABLE = False

# (defined after app creation below)

# Create FastAPI app
app = FastAPI(
    title="DiaLog API - Diabetes Meal Safety Predictor",
    description="""
    ## DiaLog API for Diabetes Meal Safety Prediction

    This API provides endpoints to:
    * Predict meal safety for diabetic users
    * Get nutritional information for foods
    * Fetch available foods from the database
    * Check API health and model status

    ### Model Information
    - Uses Random Forest Classifier trained on food nutritional data
    - Considers user BMI, sugar levels, and meal timing
    - Provides confidence scores and recommendations

    ### Usage
    1. Check `/health` to verify API is running
    2. Use `/foods` to get available food options
    3. Send meal data to `/predict` for safety analysis
    """,
    version="2.0.0",
    contact={
        "name": "DiaLog Team",
        "email": "team@dialog.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS: allow frontend origins (local + Vercel by default). Can be customized via env.
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "").split(",") if o.strip()] or _default_origins
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ai/env-check")
async def ai_env_check():
    """Return key prefix/length, selected model, cwd, and which env files were loaded."""
    key = os.getenv("GEMINI_API_KEY") or ""
    return {
        "key_prefix": key[:10],
        "key_len": len(key),
        "model": GEMINI_MODEL or "gemini-1.5-flash",
        "cwd": os.getcwd(),
        "env_files_loaded": ENV_FILES_LOADED,
    }

@app.get("/ai/key-quick")
async def ai_key_quick():
    """Minimal key validity check: attempts list_models; returns status only."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini SDK not available")
    try:
        models = genai.list_models()
        # success if iterable returns at least one item (or simply does not throw)
        count = 0
        for _ in models:
            count += 1
            if count > 0:
                break
        return {"valid": True, "sample_count": count}
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if ("api key not valid" in low) or ("api_key_invalid" in low):
            detail = "Invalid Gemini API key"
            if GEMINI_DEBUG_ERRORS:
                detail += f": {msg}"
            raise HTTPException(status_code=401, detail=detail)
        if "permission_denied" in low or "permission" in low:
            detail = "Permission denied for Generative Language API"
            if GEMINI_DEBUG_ERRORS:
                detail += f": {msg}"
            raise HTTPException(status_code=403, detail=detail)
        raise HTTPException(status_code=500, detail=f"Key check error: {msg}")

@app.get("/ai/diagnose-key")
async def ai_diagnose_key():
    """Deeper diagnostics: SDK list_models, REST list models, SDK and REST minimal generate attempts."""
    details: Dict[str, Any] = {
        "sdk_list_models": None,
        "rest_list_models": None,
        "sdk_generate": None,
        "rest_generate": None,
        "api_key_prefix": GEMINI_API_KEY[:8],
        "model": GEMINI_MODEL or "gemini-1.5-flash"
    }
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")
    model_name = GEMINI_MODEL or "gemini-2.5-flash"
    # SDK list_models
    try:
        if not GEMINI_AVAILABLE:
            details["sdk_list_models"] = {"error": "SDK not available"}
        else:
            ms = genai.list_models()
            names = []
            for m in ms:
                names.append(getattr(m, 'name', str(m)))
                if len(names) >= 5:
                    break
            details["sdk_list_models"] = {"ok": True, "sample": names}
    except Exception as e:
        details["sdk_list_models"] = {"error": str(e)}
    # REST list models (v1)
    try:
        rest_url = "https://generativelanguage.googleapis.com/v1/models"
        r = requests.get(rest_url, params={"key": GEMINI_API_KEY}, timeout=10)
        body = None
        try:
            body = r.json()
        except Exception:
            body = None
        details["rest_list_models"] = {"status": r.status_code, "body_keys": list(body.keys()) if isinstance(body, dict) else None}
        if r.status_code != 200:
            details["rest_list_models"]["error_body"] = (r.text or "")[:500]
    except Exception as e:
        details["rest_list_models"] = {"error": str(e)}
    # SDK minimal generate
    try:
        if GEMINI_AVAILABLE:
            mdl = genai.GenerativeModel(model_name)
            res = mdl.generate_content("Ping")
            txt = None
            resp_obj = getattr(res, 'response', None)
            if resp_obj is not None:
                t = getattr(resp_obj, 'text', None)
                txt = t() if callable(t) else t
            if not txt:
                t2 = getattr(res, 'text', None)
                txt = t2() if callable(t2) else t2
            details["sdk_generate"] = {"ok": True, "text_preview": (txt or "")[:120]}
        else:
            details["sdk_generate"] = {"error": "SDK not available"}
    except Exception as e:
        details["sdk_generate"] = {"error": str(e)}
    # REST minimal generate
    try:
        rest_gen_url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
        rg = requests.post(rest_gen_url, json=payload, params={"key": GEMINI_API_KEY}, timeout=12)
        if rg.status_code == 200:
            data = rg.json()
            cand = (data.get("candidates") or [{}])[0]
            part0 = (cand.get("content", {}).get("parts") or [{}])[0]
            txt = part0.get("text") or ""
            details["rest_generate"] = {"ok": True, "text_preview": txt[:120]}
        else:
            details["rest_generate"] = {"status": rg.status_code, "error_body": (rg.text or "")[:500]}
    except Exception as e:
        details["rest_generate"] = {"error": str(e)}
    # classify overall status
    overall = "partial-success"
    if details.get("sdk_generate", {}).get("ok") or details.get("rest_generate", {}).get("ok"):
        overall = "success"
    elif any((isinstance(details.get(k), dict) and details.get(k, {}).get("error") for k in ["sdk_list_models", "rest_list_models", "sdk_generate", "rest_generate"])):
        overall = "errors"
    return {"overall": overall, "diagnostics": details}

# Data and model paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ---------------- Translation service config & cache ----------------
LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com")
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", None)
TRANSLATION_CACHE_TTL = int(os.getenv("TRANSLATION_CACHE_TTL", "86400"))  # seconds, default 1 day

# simple in-memory cache: key -> {text, ts}
_translation_cache: Dict[str, Dict[str, Any]] = {}

def _tx_cache_key(text: str, src: str, tgt: str) -> str:
    h = hashlib.sha256(f"{src}|{tgt}|{text}".encode("utf-8")).hexdigest()
    return f"tx:{src}:{tgt}:{h}"

def _tx_cache_get(text: str, src: str, tgt: str) -> Optional[str]:
    k = _tx_cache_key(text, src, tgt)
    entry = _translation_cache.get(k)
    if not entry:
        return None
    if time.time() - entry["ts"] > TRANSLATION_CACHE_TTL:
        # expired
        try:
            del _translation_cache[k]
        except Exception:
            pass
        return None
    return entry["text"]

def _tx_cache_set(text: str, translated: str, src: str, tgt: str) -> None:
    k = _tx_cache_key(text, src, tgt)
    _translation_cache[k] = {"text": translated, "ts": time.time()}

def _provider_libretranslate(text: str, src: str, tgt: str) -> Optional[str]:
    try:
        url = LIBRETRANSLATE_URL.rstrip("/") + "/translate"
        payload = {
            "q": text,
            "source": src,
            "target": tgt,
            "format": "text",
        }
        if LIBRETRANSLATE_API_KEY:
            payload["api_key"] = LIBRETRANSLATE_API_KEY
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        # Some instances return string, others dict with translatedText
        if isinstance(data, dict):
            return data.get("translatedText") or data.get("translation") or None
        if isinstance(data, str):
            return data
        return None
    except Exception:
        return None

def _provider_mymemory(text: str, src: str, tgt: str) -> Optional[str]:
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {"q": text, "langpair": f"{src}|{tgt}"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 429:
            return None
        data = r.json()
        details = str(data.get("responseDetails") or "")
        if "MYMEMORY WARNING" in details:
            return None
        if data.get("responseStatus") != 200:
            return None
        return (data.get("responseData") or {}).get("translatedText")
    except Exception:
        return None

def translate_text(text: str, src: str, tgt: str) -> str:
    if not text or src == tgt:
        return text
    # cache first
    cached = _tx_cache_get(text, src, tgt)
    if cached is not None:
        return cached
    # provider chain: LibreTranslate (configurable/public) -> MyMemory -> fallback original
    for provider in (_provider_libretranslate, _provider_mymemory):
        translated = provider(text, src, tgt)
        if translated and isinstance(translated, str):
            _tx_cache_set(text, translated, src, tgt)
            return translated
    # fallback
    _tx_cache_set(text, text, src, tgt)
    return text

@app.get("/ai/models")
async def list_ai_models():
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not available")
    try:
        models = genai.list_models()
        names = []
        for m in models:
            # filter to generation-capable models if possible
            try:
                # Some SDKs expose supported_generation_methods
                methods = getattr(m, 'supported_generation_methods', None)
                if methods and ('generateContent' in methods or 'generate_content' in methods):
                    names.append(getattr(m, 'name', str(m)))
                else:
                    names.append(getattr(m, 'name', str(m)))
            except Exception:
                names.append(getattr(m, 'name', str(m)))
        return {"models": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List models error: {str(e)}")

class TranslateBatchRequest(BaseModel):
    texts: List[str]
    source: str = "en"
    target: str

class TranslateBatchResponse(BaseModel):
    translations: List[str]
class ChatMessage(BaseModel):
    role: str  # 'user' | 'model'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    text: str

@app.post("/ai/chat", response_model=ChatResponse)
async def ai_chat(req: ChatRequest):
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not available")
    try:
        # Keep conversations short server-side for cost/safety
        history = [
            {"role": m.role if m.role in ("user", "model") else "user", "parts": [{"text": m.content[:2000]}]}
            for m in (req.messages or [])[-10:]
            if m.content and isinstance(m.content, str)
        ]

        safety_preamble = (
            "You are DiaLog, a diabetes-focused assistant.\n"
            "- Provide conservative, safe dietary guidance.\n"
            "- Prefer low glycemic load options; suggest safer alternatives.\n"
            "- Avoid medical claims; include a brief non-medical advice disclaimer.\n"
            "- Be concise and friendly."
        )
        last_user = None
        for m in reversed(req.messages or []):
            if m.role == 'user':
                last_user = m.content
                break
        user_text = last_user or "Help me with DiaLog."
        prompt = f"{safety_preamble}\n\nUser: {user_text}"

        def _normalize(name: str) -> str:
            return name.split('/', 1)[-1] if name and name.startswith('models/') else name

        # Build candidate list
        candidates = []
        if GEMINI_MODEL:
            candidates.append(GEMINI_MODEL)
        # Query available models to construct a compatible list
        try:
            available = list(genai.list_models())
            # Filter to those that support generation
            def supports_gen(m):
                methods = getattr(m, 'supported_generation_methods', None)
                if not methods:
                    return True
                return ('generateContent' in methods) or ('generate_content' in methods)
            names = [getattr(m, 'name', None) for m in available if supports_gen(m)]
            names = [n for n in names if n]
            # Prefer flash variants, then pro
            flash = [n for n in names if 'flash' in n]
            pro = [n for n in names if 'pro' in n]
            others = [n for n in names if n not in flash and n not in pro]
            ordered = flash + pro + others
            # Normalize to plain ids
            candidates.extend([_normalize(n) for n in ordered])
        except Exception:
            # Fallback to some common aliases
            candidates.extend(["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-pro", "gemini-pro"]) 

        # Deduplicate preserving order
        seen = set()
        model_candidates = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                model_candidates.append(c)

        last_err = None
        for model_name in model_candidates[:6]:
            try:
                model = genai.GenerativeModel(model_name)
                try:
                    chat = model.start_chat(history=history)
                    result = chat.send_message(prompt)
                except Exception:
                    # Fallback: stateless generation
                    result = model.generate_content(prompt)
                # Extract text robustly
                text = None
                resp_obj = getattr(result, 'response', None)
                if resp_obj is not None:
                    t = getattr(resp_obj, 'text', None)
                    text = t() if callable(t) else t
                if not text:
                    t2 = getattr(result, 'text', None)
                    text = t2() if callable(t2) else t2
                if text:
                    return ChatResponse(text=text)
            except Exception as inner:
                # Map known auth/quota errors precisely
                emsg = str(inner)
                last_err = inner
                low = emsg.lower()
                if ("api key not valid" in low) or ("api_key_invalid" in low):
                    detail = "Invalid Gemini API key"
                    if GEMINI_DEBUG_ERRORS:
                        detail += f": {emsg}"
                    raise HTTPException(status_code=401, detail=detail)
                if ("permission" in low) or ("permission_denied" in low):
                    detail = "Permission denied for Gemini model"
                    if GEMINI_DEBUG_ERRORS:
                        detail += f": {emsg}"
                    raise HTTPException(status_code=403, detail=detail)
                if ("quota" in low) or ("rate" in low) or ("429" in low):
                    raise HTTPException(status_code=429, detail="Gemini quota exceeded")
                # else continue to next candidate
                continue
        # If SDK attempts failed on all candidates, try REST fallback once with the first candidate
        try:
            fallback_model = model_candidates[0] if model_candidates else (GEMINI_MODEL or "gemini-2.5-flash")
            rest_url = f"https://generativelanguage.googleapis.com/v1/models/{fallback_model}:generateContent"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(rest_url, json=payload, params={"key": GEMINI_API_KEY}, timeout=18)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates") or []
                if candidates:
                    content = candidates[0].get("content") or {}
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], dict):
                        txt = parts[0].get("text")
                        if txt:
                            return ChatResponse(text=txt)
            elif r.status_code == 401:
                detail = "Invalid Gemini API key (REST)"
                if GEMINI_DEBUG_ERRORS:
                    detail += f" body={r.text[:200]}"
                raise HTTPException(status_code=401, detail=detail)
            elif r.status_code == 403:
                detail = "Gemini permission denied (REST)"
                if GEMINI_DEBUG_ERRORS:
                    detail += f" body={r.text[:200]}"
                raise HTTPException(status_code=403, detail=detail)
            elif r.status_code in (429,):
                raise HTTPException(status_code=429, detail="Gemini quota exceeded")
        except HTTPException:
            raise
        except Exception as rest_e:
            last_err = rest_e
        # If all models failed, return a graceful message and expose the first few candidates to help config
        raise HTTPException(status_code=502, detail=f"AI model unavailable. Tried: {model_candidates[:6]}. Last error: {last_err}")
    except Exception as e:
        # Preserve mapped HTTPExceptions
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")



# Endpoint to log each meal in the list to Firestore
@app.post("/log-meal-firestore")
async def log_meal_to_firestore(log: MealLog):
    if not FIREBASE_AVAILABLE or not firebase_initialized or not firestore_db:
        raise HTTPException(status_code=503, detail="Firebase/Firestore not available")
    
    try:
        results = []
        for meal in log.meals:
            # Prepare prediction request for each meal with default values
            predict_req = MealRequest(
                age=35,  # Default age
                gender="Male",  # Default gender
                weight_kg=70,  # Default weight in kg
                height_cm=170,  # Default height in cm
                fasting_sugar=log.sugar_level_fasting,
                post_meal_sugar=log.sugar_level_post,
                meal_taken=meal.meal_name,
                time_of_day=meal.time_of_day,
                portion_size=meal.quantity,
                portion_unit=meal.unit
            )
            prediction = await predict_meal_safety(predict_req)
            # Prepare log entry
            log_entry = {
                "userId": log.userId,
                "meal_name": meal.meal_name,
                "quantity": meal.quantity,
                "unit": meal.unit,
                "time_of_day": meal.time_of_day,
                "sugar_level_fasting": log.sugar_level_fasting,
                "sugar_level_post": log.sugar_level_post,
                "prediction": prediction.dict(),
                "createdAt": firestore.SERVER_TIMESTAMP if not log.createdAt else log.createdAt
            }
            if FIREBASE_AVAILABLE and firestore_db:
                doc_ref = firestore_db.collection("logs").add(log_entry)
            results.append({"doc_id": doc_ref[1].id, "meal": meal.meal_name, "risk": prediction.risk_level})
        # Calculate overall risk for the meal event
        risk_levels = [r["risk"] for r in results]
        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        summary_entry = {
            "userId": log.userId,
            "meals": [r["meal"] for r in results],
            "sugar_level_fasting": log.sugar_level_fasting,
            "sugar_level_post": log.sugar_level_post,
            "overall_risk": overall_risk,
            "individual_risks": risk_levels,
            "createdAt": firestore.SERVER_TIMESTAMP if not log.createdAt else log.createdAt
        }
        if FIREBASE_AVAILABLE and firestore_db:
            firestore_db.collection("logs_summary").add(summary_entry)

        return {"success": True, "results": results, "overall_risk": overall_risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log meals: {str(e)}")

# Load the food dataset directly
try:
    food_df = pd.read_csv(DATA_DIR / "Food_Master_Dataset_.csv")
    print(f"Loaded {len(food_df)} foods from dataset")
    food_df.set_index('dish_name', inplace=True)
except Exception as e:
    print(f"Error loading food dataset: {e}")
    food_df = pd.DataFrame()

# Global variables for model artifacts
model = None
scaler = None
feature_columns = None
# New improved predictor
meal_safety_predictor = None
# Personalized ML recommender
personalized_recommender = None

# Load model and artifacts
def load_model_artifacts():
    global model, scaler, feature_columns, meal_safety_predictor, personalized_recommender
    try:
        # Initialize improved prediction system with medical model
        meal_safety_predictor = MealSafetyPredictor()
        meal_safety_predictor.load_food_dataset(DATA_DIR / "Food_Master_Dataset_.csv")
        meal_safety_predictor.load_model(MODEL_DIR)  # This will load the medical model
        
        print("✅ Medical prediction system initialized")
        
        # Initialize personalized ML recommender
        if PERSONALIZED_ML_AVAILABLE:
            try:
                personalized_recommender = PersonalizedMealRecommender(
                    data_path=str(DATA_DIR / "User_Logs_Dataset.csv")
                )
                print("✅ Personalized ML recommender initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize personalized recommender: {e}")
                personalized_recommender = None
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Pydantic models for request/response
class MealRequest(BaseModel):
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    fasting_sugar: float
    post_meal_sugar: float
    meal_taken: str
    time_of_day: str
    portion_size: float
    portion_unit: str
    # Optional personalization fields
    user_id: Optional[int] = None
    diabetes_type: Optional[str] = None

class MealItem(BaseModel):
    meal_taken: str
    portion_size: float
    portion_unit: str

class MultipleMealRequest(BaseModel):
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    fasting_sugar: float
    post_meal_sugar: float
    time_of_day: str
    meals: List[MealItem]

class NutritionalInfo(BaseModel):
    calories: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float

class Recommendation(BaseModel):
    name: str
    reason: str

class PredictionResponse(BaseModel):
    is_safe: bool
    confidence: float
    risk_level: str
    message: str
    bmi: float
    nutritional_info: Optional[NutritionalInfo] = None
    recommendations: Optional[List[Recommendation]] = None
    # Additional safety context
    glycemic_load: Optional[float] = None
    personalized_predicted_blood_sugar: Optional[float] = None
    model_used: Optional[str] = None
    # Color-coded badges for UI
    risk_badge: Optional[Dict[str, Any]] = None
    gl_badge: Optional[Dict[str, Any]] = None

class IndividualMealPrediction(BaseModel):
    meal: str
    is_safe: bool
    confidence: float
    risk_level: str
    nutritional_info: Optional[NutritionalInfo] = None

class MultipleMealResponse(BaseModel):
    is_safe: bool
    overall_risk_level: str
    message: str
    bmi: float
    confidence: float
    predictions: List[IndividualMealPrediction]
    recommendations: Optional[List[Recommendation]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    foods_count: int
    version: str

class FoodsResponse(BaseModel):
    foods: List[str]
    count: int

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_artifacts()

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Welcome to DiaLog API - Diabetes Meal Safety Predictor",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        foods_count=len(food_df),
        version="2.0.0"
    )

@app.get("/foods", response_model=FoodsResponse)
async def get_foods(search: Optional[str] = Query(None, description="Search term to filter foods")):
    try:
        if food_df.empty:
            raise HTTPException(status_code=500, detail="Food database not loaded")
        
        foods_list = food_df.index.tolist()
        
        if search:
            search_lower = search.lower()
            foods_list = [food for food in foods_list if search_lower in food.lower()]
        
        return FoodsResponse(foods=sorted(foods_list), count=len(foods_list))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching foods: {str(e)}")

# ---------------- Translation endpoints ----------------
@app.post("/translate-batch", response_model=TranslateBatchResponse)
async def translate_batch(req: TranslateBatchRequest):
    try:
        if not req.texts:
            return TranslateBatchResponse(translations=[])
        # Deduplicate to cut provider calls
        unique = list(dict.fromkeys(req.texts))
        mapped: Dict[str, str] = {}
        for t in unique:
            mapped[t] = translate_text(t, req.source, req.target)
        # map in original order
        out = [mapped.get(t, t) for t in req.texts]
        return TranslateBatchResponse(translations=out)
    except Exception as e:
        # On any error, gracefully fall back to originals
        return TranslateBatchResponse(translations=[t for t in req.texts])

@app.get("/translate")
async def translate(text: str, source: str = "en", target: str = "hi"):
    try:
        return {"translation": translate_text(text, source, target)}
    except Exception:
        return {"translation": text}

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    # Safety check to prevent division by zero
    if height_cm <= 0 or weight_kg <= 0:
        return 25.0  # Return normal BMI as default
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        v = float(val)
        if v != v:  # NaN check
            return default
        return v
    except Exception:
        return default

def _compute_gl_for_standard_portion(food_row: pd.Series, portion_g: float = 200.0) -> Optional[float]:
    """Compute glycemic load for a given portion based on available columns.
    Prefers explicit glycemic_load column; else uses GI and carbs if available.
    Returns None if insufficient data.
    """
    # Direct glycemic_load per 100g
    if 'glycemic_load' in food_row.index:
        gl_per100 = _safe_float(food_row.get('glycemic_load'), None)
        if gl_per100 is not None:
            return gl_per100 * (portion_g / 100.0)
    # Alternate common column names
    if 'Glycemic Load' in food_row.index:
        gl_per100 = _safe_float(food_row.get('Glycemic Load'), None)
        if gl_per100 is not None:
            return gl_per100 * (portion_g / 100.0)
    # Compute from GI and carbs
    gi = None
    for key in ['glycemic_index', 'GI', 'gi']:
        if key in food_row.index:
            gi = _safe_float(food_row.get(key), None)
            break
    carbs_g_per100 = None
    for key in ['carbs_g', 'Carbohydrate (g)', 'carbohydrates_g']:
        if key in food_row.index:
            carbs_g_per100 = _safe_float(food_row.get(key), None)
            break
    if gi is None or carbs_g_per100 is None:
        return None
    carbs_for_portion = carbs_g_per100 * (portion_g / 100.0)
    # GL formula: (GI * carbs_g)/100
    return (gi * carbs_for_portion) / 100.0

def _gl_threshold_by_diabetes(diabetes_type: Optional[str]) -> float:
    """Return GL threshold for a portion based on diabetes type.
    Defaults to 15.0 when unknown.
    Gestational: 10.0, Prediabetes: 12.0, Type1: 15.0, Type2: 15.0
    """
    dt = (diabetes_type or "").strip().lower()
    if "gestational" in dt:
        return 10.0
    if "prediabetes" in dt or "pre-diabetes" in dt:
        return 12.0
    if "type1" in dt or "type 1" in dt:
        return 15.0
    if "type2" in dt or "type 2" in dt:
        return 15.0
    return 15.0

def _gl_universal_cutoff() -> float:
    return 15.0

def _gl_badge(gl_value: Optional[float], cutoff: float = 15.0) -> Optional[Dict[str, Any]]:
    try:
        if gl_value is None:
            return None
        val = float(gl_value)
        color = "green" if val < 10.0 else ("yellow" if val < cutoff else "red")
        return {"label": f"GL {val:.1f}", "color": color, "value": val}
    except Exception:
        return None

# -------- Veg/Non-veg helpers --------
NON_VEG_KEYWORDS = {
    # Common meats
    'chicken','mutton','fish','egg','meat','prawn','prawns','shrimp','crab','keema','kebab','kebabs','tikka','boti','paya',
    'lamb','beef','pork','turkey','duck','seafood','salmon','tuna','sardine','anchovy','octopus','squid','ham','salami','pepperoni','prosciutto','bacon','sausage',
    # Phrases and Indian variants
    'biryani chicken','chicken curry','fish curry','tandoori','seekh','shami','galouti','nihari','bhuna chicken','butter chicken',
    # Egg forms
    'egg curry','omelet','omelette','scrambled egg','boiled egg','egg bhurji','anda bhurji','anda','murgh','murg'
}

def is_vegetarian_name(food_name: str) -> bool:
    try:
        n = (food_name or '').lower()
        # If any explicit non-veg token is present, it's non-veg regardless of labels
        if any(k in n for k in NON_VEG_KEYWORDS):
            # Allow only explicit 'eggless' to override 'egg'
            if 'egg' in n and 'eggless' in n:
                # proceed to other checks
                pass
            else:
                return False
        # Explicit eggless indicates vegetarian
        if 'eggless' in n:
            return True
        # Hints toward vegetarian
        if any(h in n for h in ['veg ', '(veg', '[veg', 'pure veg', 'vegetarian']):
            return True
        # Default: no non-veg token means vegetarian
        return True
    except Exception:
        return True

def get_nutritional_info_enhanced(food_name: str, portion_size_g: float, 
                                portion_features: Dict[str, float]) -> NutritionalInfo:
    """
    Enhanced nutritional info that includes portion-adjusted values.
    """
    if food_name not in food_df.index:
        return None
    
    # Use portion-aware features for more accurate info
    return NutritionalInfo(
        calories=portion_features['calories_effective_kcal'],
        carbs_g=portion_features['carbs_effective_g'],
        protein_g=food_df.loc[food_name].get('protein_g', 0) * portion_features['portion_multiplier'],
        fat_g=food_df.loc[food_name].get('fat_g', 0) * portion_features['portion_multiplier'],
        fiber_g=food_df.loc[food_name].get('fiber_g', 0) * portion_features['portion_multiplier']
    )

def generate_enhanced_recommendations(food_name: str, prediction_result: Dict[str, any], 
                                   bmi: float, user_data: Dict[str, any]) -> List[Recommendation]:
    """
    Generate recommendations based on the improved prediction system.
    """
    recommendations = []
    risk_level = prediction_result['risk_level']
    reasons = prediction_result.get('reasons', [])
    portion_features = prediction_result.get('portion_features', {})
    
    # Risk-specific recommendations
    if risk_level == 'unsafe':
        recommendations.append(Recommendation(
            name="Avoid This Meal",
            reason="Multiple risk factors detected. Consider alternatives or significantly reduce portion."
        ))
        
        if portion_features.get('portion_multiplier', 1) > 2:
            recommendations.append(Recommendation(
                name="Reduce Portion Size", 
                reason=f"Current portion is {portion_features['portion_multiplier']:.1f}× normal. Try 0.5-1× instead."
            ))
            
        if portion_features.get('GL_portion', 0) > 20:
            recommendations.append(Recommendation(
                name="Add Fiber and Protein",
                reason="High glycemic load. Pair with vegetables and protein to slow absorption."
            ))
            
    elif risk_level == 'caution':
        recommendations.append(Recommendation(
            name="Monitor Closely",
            reason="Some risk factors present. Check blood sugar 2 hours after eating."
        ))
        
        if portion_features.get('sugar_effective_g', 0) > 25:
            recommendations.append(Recommendation(
                name="Post-Meal Walk",
                reason=f"High sugar content ({portion_features['sugar_effective_g']:.0f}g). Walk for 15-20 minutes."
            ))
            
    else:  # safe
        recommendations.append(Recommendation(
            name="Good Choice",
            reason="This meal appears suitable for your profile. Continue monitoring as usual."
        ))
    
    # BMI-specific advice
    if bmi > 25:
        recommendations.append(Recommendation(
            name="Portion Control",
            reason="Focus on portion sizes to support healthy weight management."
        ))
    
    # Always add general diabetes advice
    recommendations.append(Recommendation(
        name="Post-Meal Activity",
        reason="Light physical activity after meals helps regulate blood sugar."
    ))
    
    return recommendations

def get_nutritional_info(food_name: str, portion_size: float) -> NutritionalInfo:
    if food_name not in food_df.index:
        return None
    
    food_row = food_df.loc[food_name]
    
    # Calculate nutritional values based on portion size
    # Assuming the dataset values are per 100g serving
    multiplier = portion_size / 1.0  # Adjust based on your portion unit logic
    
    return NutritionalInfo(
        calories=float(food_row.get('calories_kcal', 0)) * multiplier,
        carbs_g=float(food_row.get('carbs_g', 0)) * multiplier,
        protein_g=float(food_row.get('protein_g', 0)) * multiplier,
        fat_g=float(food_row.get('fat_g', 0)) * multiplier,
        fiber_g=float(food_row.get('fiber_g', 0)) * multiplier
    )

def generate_recommendations(food_name: str, is_safe: bool, bmi: float) -> List[Recommendation]:
    recommendations = []
    
    if not is_safe:
        recommendations.append(Recommendation(
            name="Portion Control",
            reason="Consider reducing portion size by 25-30% to minimize blood sugar impact"
        ))
        
        recommendations.append(Recommendation(
            name="Add Fiber",
            reason="Include high-fiber vegetables or salad to slow sugar absorption"
        ))
    
    if bmi > 25:
        recommendations.append(Recommendation(
            name="Weight Management",
            reason="Consider lower calorie alternatives to support healthy weight"
        ))
    
    # Add exercise recommendation
    recommendations.append(Recommendation(
        name="Post-meal Activity",
        reason="Take a 10-15 minute walk after eating to help regulate blood sugar"
    ))
    
    return recommendations

@app.post("/predict", response_model=PredictionResponse)
async def predict_meal_safety(request: MealRequest):
    """
    Improved meal safety prediction with hard guardrails and portion awareness.
    """
    try:
        global meal_safety_predictor
        
        if meal_safety_predictor is None:
            raise HTTPException(status_code=503, detail="Prediction system not initialized")
        
        # Convert portion unit to grams (simplified conversion)
        portion_unit_to_grams = {
            'cup': 200, 'bowl': 250, 'plate': 300, 'piece': 100,
            'slice': 50, 'spoon': 15, 'glass': 250, 'g': 1, 'grams': 1
        }
        portion_size_g = request.portion_size * portion_unit_to_grams.get(request.portion_unit.lower(), 100)
        
        # Calculate BMI
        bmi = calculate_bmi(request.weight_kg, request.height_cm)
        
        # Prepare user context for prediction
        user_data = {
            'age': request.age,
            'gender': request.gender,
            'bmi': bmi,
            'fasting_sugar': request.fasting_sugar,
            'post_meal_sugar': request.post_meal_sugar,
            'time_of_day': request.time_of_day
        }
        
        # Use improved prediction system
        result = meal_safety_predictor.predict_meal_safety(
            request.meal_taken, 
            portion_size_g, 
            user_data
        )
        
        # Map risk levels to expected format
        risk_mapping = {
            'safe': ('low', True),
            'caution': ('medium', False), 
            'unsafe': ('high', False)
        }
        
        risk_level, is_safe = risk_mapping.get(result['risk_level'], ('medium', False))
        
        # Create response message with explanation
        message = result['explanation']
        confidence = result['confidence']

        # Strict GL threshold per actual portion by diabetes type
        gl_portion = None
        try:
            gl_portion = result.get('portion_features', {}).get('GL_portion')
        except Exception:
            gl_portion = None
        if gl_portion is not None:
            try:
                gl_cutoff = _gl_universal_cutoff()
                if float(gl_portion) >= gl_cutoff:
                    risk_level = 'high'
                    is_safe = False
                    # Keep explanation neutral without exposing numeric thresholds
                    message = "This portion’s glycemic impact appears high for your profile. Prefer a smaller portion or choose an alternative."
            except Exception:
                pass

        # Optional personalized override using user's model if available
        personalized_pred = None
        model_used = None
        try:
            if personalized_recommender is not None and request.user_id is not None:
                # Build features consistent with personalized model
                user_features_p = {
                    'Age': request.age,
                    'Weight': request.weight_kg,
                    'Height': request.height_cm,
                    'BMI': bmi,
                    'Gender_encoded': 1 if request.gender.lower() == 'male' else 0,
                    'Diabetes_Type_encoded': 0 if (request.diabetes_type or '').lower() == 'type1' else 1,
                    'Meal_Time_encoded': {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Snack': 3}.get(request.time_of_day, 1)
                }
                personalized_pred = personalized_recommender.predict_blood_sugar(
                    request.user_id, request.meal_taken, user_features_p
                )
                # Determine model used
                model_used = 'general'
                try:
                    if hasattr(personalized_recommender, 'user_models') and request.user_id in personalized_recommender.user_models:
                        model_used = 'personal'
                    else:
                        dtype = (request.diabetes_type or '').strip()
                        if hasattr(personalized_recommender, 'general_models_by_diabetes'):
                            keys = list(getattr(personalized_recommender, 'general_models_by_diabetes', {}).keys())
                            if any(k.lower() == dtype.lower() for k in keys):
                                model_used = 'cohort'
                except Exception:
                    pass

                # Apply conservative thresholds by diabetes type
                dtype = (request.diabetes_type or 'Type2').lower()
                if dtype == 'gestational':
                    safe_thr, caution_thr = 120, 140
                elif dtype == 'prediabetes':
                    safe_thr, caution_thr = 130, 160
                elif dtype == 'type1':
                    safe_thr, caution_thr = 140, 180
                else:  # type2 or others
                    safe_thr, caution_thr = 140, 170

                # Override risk conservatively based on personalized prediction
                try:
                    pb = float(personalized_pred)
                    if pb > caution_thr:
                        risk_level = 'high'
                        is_safe = False
                        message = f"Personalized prediction {pb:.0f} mg/dL exceeds {caution_thr} for {request.diabetes_type or 'Type2'}. Avoid or choose alternative."
                    elif pb > safe_thr:
                        # At least caution
                        # If already unsafe from GL, keep unsafe
                        if risk_level != 'high':
                            risk_level = 'medium'
                            is_safe = False
                            message = f"Personalized prediction {pb:.0f} mg/dL above safe threshold {safe_thr}. If consumed, use strict portion control and monitor."
                except Exception:
                    pass
        except Exception:
            # Personalization should never break baseline safety
            pass
        
        # Get nutritional information (enhanced with portion awareness)
        nutritional_info = get_nutritional_info_enhanced(
            request.meal_taken, 
            portion_size_g, 
            result['portion_features']
        )
        
        # Generate enhanced recommendations based on guardrails
        recommendations = generate_enhanced_recommendations(
            request.meal_taken, 
            result, 
            bmi,
            user_data
        )
        
        # Build badges
        def _risk_badge(level: str) -> Dict[str, Any]:
            lvl = (level or '').lower()
            if lvl == 'high':
                return {"label": "UNSAFE", "color": "red"}
            if lvl == 'medium' or lvl == 'moderate':
                return {"label": "CAUTION", "color": "yellow"}
            return {"label": "SAFE", "color": "green"}

        gl_badge = _gl_badge(gl_portion, _gl_universal_cutoff()) if gl_portion is not None else None

        return PredictionResponse(
            is_safe=is_safe,
            confidence=confidence,
            risk_level=risk_level,
            message=message,
            bmi=bmi,
            nutritional_info=nutritional_info,
            recommendations=recommendations,
            glycemic_load=(float(gl_portion) if gl_portion is not None else None),
            personalized_predicted_blood_sugar=(float(personalized_pred) if personalized_pred is not None else None),
            model_used=model_used,
            risk_badge=_risk_badge(risk_level),
            gl_badge=gl_badge
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-multiple", response_model=MultipleMealResponse)
async def predict_multiple_meals(request: MultipleMealRequest):
    """
    Predict safety for multiple meals consumed at the same time.
    """
    try:
        global meal_safety_predictor
        
        if meal_safety_predictor is None:
            raise HTTPException(status_code=503, detail="Prediction system not initialized")
        
        # Convert portion unit to grams mapping
        portion_unit_to_grams = {
            'cup': 200, 'bowl': 250, 'plate': 300, 'piece': 100,
            'slice': 50, 'spoon': 15, 'glass': 250, 'g': 1, 'grams': 1
        }
        
        # Calculate BMI
        bmi = calculate_bmi(request.weight_kg, request.height_cm)
        
        # Prepare user context
        user_data = {
            'age': request.age,
            'gender': request.gender,
            'bmi': bmi,
            'fasting_sugar': request.fasting_sugar,
            'post_meal_sugar': request.post_meal_sugar,
            'time_of_day': request.time_of_day
        }
        
        # Process each meal
        individual_predictions = []
        overall_risk_scores = []
        is_any_unsafe = False
        
        for meal_item in request.meals:
            # Convert portion to grams
            portion_size_g = meal_item.portion_size * portion_unit_to_grams.get(meal_item.portion_unit.lower(), 100)
            
            # Get prediction for this meal
            result = meal_safety_predictor.predict_meal_safety(
                meal_item.meal_taken, 
                portion_size_g, 
                user_data
            )
            
            # Map risk levels
            risk_mapping = {
                'safe': ('low', True),
                'caution': ('moderate', False), 
                'unsafe': ('high', False)
            }
            
            risk_level, is_safe = risk_mapping.get(result['risk_level'], ('moderate', False))
            
            if not is_safe:
                is_any_unsafe = True
            
            # Store risk score for overall calculation
            risk_scores = {'low': 0, 'moderate': 1, 'high': 2}
            overall_risk_scores.append(risk_scores.get(risk_level, 1))
            
            # Get nutritional info
            nutritional_info = get_nutritional_info_enhanced(
                meal_item.meal_taken, 
                portion_size_g, 
                result['portion_features']
            )
            
            individual_predictions.append(IndividualMealPrediction(
                meal=meal_item.meal_taken,
                is_safe=is_safe,
                confidence=result['confidence'],
                risk_level=risk_level,
                nutritional_info=nutritional_info
            ))
        
        # Determine overall risk level
        avg_risk_score = sum(overall_risk_scores) / len(overall_risk_scores)
        if avg_risk_score >= 1.5:
            overall_risk_level = 'high'
        elif avg_risk_score >= 0.5:
            overall_risk_level = 'moderate'
        else:
            overall_risk_level = 'low'
        
        # Calculate overall confidence
        overall_confidence = sum(pred.confidence for pred in individual_predictions) / len(individual_predictions)
        
        # Generate combined recommendations for the meal combination
        combined_recommendations = []
        high_risk_meals = [pred.meal for pred in individual_predictions if pred.risk_level == 'high']
        moderate_risk_meals = [pred.meal for pred in individual_predictions if pred.risk_level == 'moderate']
        
        if high_risk_meals:
            combined_recommendations.append(Recommendation(
                name="High Risk Items in Combination",
                reason=f"Consider replacing or reducing portions of: {', '.join(high_risk_meals)} in your meal combination"
            ))
        
        if overall_risk_level == 'high':
            combined_recommendations.append(Recommendation(
                name="High Risk Meal Combination",
                reason="This overall meal combination poses high risk. Consider eating these items at different times or reducing portions significantly."
            ))
        elif overall_risk_level == 'moderate':
            combined_recommendations.append(Recommendation(
                name="Moderate Risk Meal Combination",
                reason="Monitor blood sugar levels closely after this meal combination. Consider spacing out high-carb items."
            ))
        
        # Add combination-specific advice
        if len(request.meals) > 2:
            combined_recommendations.append(Recommendation(
                name="Multiple Item Strategy",
                reason=f"When eating {len(request.meals)} items together, consider eating protein-rich items first to slow glucose absorption."
            ))
        
        # Create response message
        meal_count = len(request.meals)
        meal_list = ", ".join([meal.meal_taken for meal in request.meals[:3]])  # Show first 3 meals
        if meal_count > 3:
            meal_list += f" and {meal_count - 3} more items"
        
        if is_any_unsafe:
            message = f"Combined analysis for {meal_list}: This meal combination may need attention due to elevated risk."
        elif overall_risk_level == 'moderate':
            message = f"Combined analysis for {meal_list}: Moderate risk detected. Monitor blood sugar levels after eating."
        else:
            message = f"Combined analysis for {meal_list}: This meal combination appears safe for your diabetes management."
        
        return MultipleMealResponse(
            is_safe=not is_any_unsafe,
            overall_risk_level=overall_risk_level,
            message=message,
            bmi=bmi,
            confidence=overall_confidence,
            predictions=individual_predictions,
            recommendations=combined_recommendations if combined_recommendations else None
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple meal prediction error: {str(e)}")

def _to_native(value: Any):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    try:
        import numpy as _np
        if value is None:
            return None
        if isinstance(value, (_np.floating, _np.float32, _np.float64)):
            v = float(value)
            return None if (v != v) else v  # handle NaN
        if isinstance(value, (_np.integer,)):
            return int(value)
    except Exception:
        pass
    # Handle pandas NA/NaT
    try:
        import pandas as _pd
        if _pd.isna(value):
            return None
    except Exception:
        pass
    # Cast common numeric strings
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
    except Exception:
        pass
    return value

@app.get("/food/{food_name}")
async def get_food_details(food_name: str):
    try:
        if food_df.empty:
            raise HTTPException(status_code=500, detail="Food database not loaded")
        
        if food_name not in food_df.index:
            raise HTTPException(status_code=404, detail="Food not found")
        
        food_row = food_df.loc[food_name]

        # Build a comprehensive nutrition dict from known columns if present
        nutrition_keys = [
            'calories_kcal', 'carbs_g', 'protein_g', 'fat_g', 'fiber_g', 'sugar_g',
            'glycemic_index', 'glycemic_load', 'sodium_mg', 'serving_size_g',
            'default_portion'
        ]
        nutritional_info = {}
        for k in nutrition_keys:
            if k in food_row.index:
                nutritional_info[k] = _to_native(food_row.get(k))

        # Safety metadata
        safety_info = {
            "avoid_for_diabetic": _to_native(food_row.get('avoid_for_diabetic', 'No')),
            "safe_threshold_sugar": _to_native(food_row.get('safe_threshold_sugar', 110)),
            "risky_threshold_sugar": _to_native(food_row.get('risky_threshold_sugar', 140)),
            "risky_reason": _to_native(food_row.get('risky_reason')),
            "recommended_alternatives": _to_native(food_row.get('recommended_alternatives')),
        }

        # Additional descriptive fields
        descriptors = {}
        for k in ['food_id','food_type','cuisine_region','meal_time_category','meal_time_fit','vitamins_minerals_info','dietitian_notes','portion_adjustment']:
            if k in food_row.index:
                descriptors[k] = _to_native(food_row.get(k))

        # Full row as key -> value (stringified keys as-is)
        raw = { str(k): _to_native(v) for k, v in food_row.to_dict().items() }

        # Backward-compatible top-level shortcuts expected by older UI
        shortcuts = {
            "calories": nutritional_info.get('calories_kcal'),
            "carbs": nutritional_info.get('carbs_g'),
            "protein": nutritional_info.get('protein_g'),
            "fat": nutritional_info.get('fat_g'),
            "fiber": nutritional_info.get('fiber_g'),
            "glycemicIndex": nutritional_info.get('glycemic_index'),
            "glycemic_load": nutritional_info.get('glycemic_load'),
        }

        return {
            "name": food_name,
            "nutritional_info": nutritional_info,
            "safety_info": safety_info,
            "descriptors": descriptors,
            "raw": raw,
            **shortcuts
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching food details: {str(e)}")

@app.get("/test-guardrails")
async def test_guardrails():
    """
    Run acceptance tests for the improved prediction system.
    """
    try:
        global meal_safety_predictor
        
        if meal_safety_predictor is None:
            raise HTTPException(status_code=503, detail="Prediction system not initialized")
        
        # Run the acceptance tests
        test_results = run_acceptance_tests(meal_safety_predictor)
        
        return {
            "success": True,
            "test_results": test_results,
            "message": f"Tests completed: {test_results['passed']}/{test_results['passed'] + test_results['failed']} passed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

@app.get("/predict-sample")
async def predict_sample():
    """
    Sample prediction to demonstrate the improved system.
    """
    try:
        global meal_safety_predictor
        
        if meal_safety_predictor is None:
            raise HTTPException(status_code=503, detail="Prediction system not initialized")
        
        # Sample cases showing the improvements
        sample_cases = [
            {
                "case": "Normal portion of safe food",
                "meal": "Hot tea (Garam Chai)",
                "portion_g": 200,
                "user": {"age": 45, "gender": "Male", "bmi": 25, "fasting_sugar": 100, "time_of_day": "Breakfast"}
            },
            {
                "case": "Large portion triggering guardrails",
                "meal": "Plain cream cake", 
                "portion_g": 150,
                "user": {"age": 45, "gender": "Male", "bmi": 25, "fasting_sugar": 100, "time_of_day": "Snack"}
            }
        ]
        
        results = []
        for case in sample_cases:
            try:
                prediction = meal_safety_predictor.predict_meal_safety(
                    case["meal"], case["portion_g"], case["user"]
                )
                results.append({
                    "case": case["case"],
                    "meal": case["meal"], 
                    "risk_level": prediction["risk_level"],
                    "explanation": prediction["explanation"]
                })
            except Exception as e:
                results.append({
                    "case": case["case"],
                    "error": str(e)
                })
        
        return {"sample_predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample prediction error: {str(e)}")

class PersonalizedRecommendationRequest(BaseModel):
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    fasting_sugar: int
    post_meal_sugar: int
    diabetes_type: str
    time_of_day: str = "Lunch"
    meal_preferences: Optional[List[str]] = None
    count: int = 6

@app.post("/recommendations")
async def get_personalized_recommendations(request: PersonalizedRecommendationRequest):
    """
    Generate truly personalized meal recommendations using ML model.
    """
    if meal_safety_predictor is None:
        raise HTTPException(status_code=503, detail="Prediction system not initialized")
    
    try:
        # Calculate BMI for context
        bmi = calculate_bmi(request.weight_kg, request.height_cm)
        
        # Create user context
        user_data = {
            'age': request.age,
            'gender': request.gender,
            'bmi': bmi,
            'fasting_sugar': request.fasting_sugar,
            'post_meal_sugar': request.post_meal_sugar,
            'time_of_day': request.time_of_day,
            'diabetes_type': request.diabetes_type
        }
        
        # Get all available foods
        if not hasattr(meal_safety_predictor, 'food_df') or meal_safety_predictor.food_df is None:
            raise HTTPException(status_code=503, detail="Food dataset not loaded")
        
        all_foods = list(meal_safety_predictor.food_df.index)
        
        # Filter foods based on time of day and preferences
        time_filters = {
            'Breakfast': ['idli', 'dosa', 'poha', 'upma', 'oats', 'daliya', 'paratha'],
            'Lunch': ['dal', 'rice', 'roti', 'sabzi', 'curry', 'pulao', 'khichdi'],
            'Dinner': ['soup', 'dal', 'roti', 'sabzi', 'curry', 'vegetable'],
            'Snack': ['fruit', 'nuts', 'tea', 'milk', 'sprouts', 'chaat']
        }
        
        relevant_keywords = time_filters.get(request.time_of_day, time_filters['Lunch'])
        
        # Find foods matching time of day
        candidate_foods = []
        for food in all_foods:
            fl = food.lower()
            if any(keyword in fl for keyword in relevant_keywords):
                candidate_foods.append(food)

        # Apply vegetarian preference if requested
        if request.meal_preferences and any(p.lower() == 'vegetarian' for p in request.meal_preferences):
            candidate_foods = [f for f in candidate_foods if is_vegetarian_name(f)]

        # If nothing matched after filters, broaden to time-of-day only (no veg filter relaxation)
        if not candidate_foods:
            candidate_foods = [f for f in all_foods if any(k in f.lower() for k in relevant_keywords)]
        
        # Test each food with ML model and rank by safety
        food_scores = []
        
        for food in candidate_foods[:50]:  # Test up to 50 foods for performance
            try:
                # Use standard portion size for comparison
                standard_portion = 200  # 200g standard
                
                prediction = meal_safety_predictor.predict_meal_safety(
                    food, standard_portion, user_data
                )
                
                # Calculate safety score (higher = safer). If not 'safe', mark very low to be filtered later
                risk_scores = {'safe': 1.0, 'caution': 0.01, 'unsafe': 0.0}
                safety_score = risk_scores.get(prediction['risk_level'], 0.0)
                confidence = prediction['confidence']
                
                # Combined score weighted by confidence
                final_score = safety_score * confidence
                
                food_scores.append({
                    'food_name': food,
                    'safety_score': final_score,
                    'risk_level': prediction['risk_level'],
                    'confidence': confidence,
                    'explanation': prediction['explanation'],
                    'portion_features': prediction.get('portion_features', {}),
                    'reasons': prediction.get('reasons', [])
                })
                
            except Exception as e:
                # Skip foods that cause errors
                continue
        
        # Filter out foods with glycemic load above threshold for standard portion (per diabetes type)
        gl_cutoff = _gl_universal_cutoff()
        filtered_scores = []
        for item in food_scores:
            try:
                row = meal_safety_predictor.food_df.loc[item['food_name']]
                gl200 = _compute_gl_for_standard_portion(row, 200.0)
                # If GL unknown or >= cutoff, skip conservatively
                if gl200 is None or gl200 >= gl_cutoff:
                    continue
                filtered_scores.append(item)
            except Exception:
                # If any error computing GL, be conservative and skip
                continue
        # Keep only low-risk (safe) items
        safe_only = [it for it in filtered_scores if it.get('risk_level') == 'safe']
        # Sort by safety score (highest first) and select top recommendations
        safe_only.sort(key=lambda x: x['safety_score'], reverse=True)
        top_recommendations = safe_only[:request.count]
        
        # Generate dynamic reasons for each recommendation
        recommendations = []
        for food_rec in top_recommendations:
            food_row = meal_safety_predictor.food_df.loc[food_rec['food_name']]
            
            # Generate intelligent, food-specific reasons
            dynamic_reasons = generate_intelligent_reasons(
                food_rec['food_name'], 
                food_row, 
                food_rec['portion_features'],
                user_data,
                food_rec['risk_level']
            )
            
            # Skip if GL>cutoff for 200g portion
            gl200 = _compute_gl_for_standard_portion(food_row, 200.0)
            if gl200 is None or gl200 >= gl_cutoff:
                continue

            recommendations.append({
                'name': food_rec['food_name'],
                'risk_level': food_rec['risk_level'],
                'confidence': food_rec['confidence'],
                'safety_score': food_rec['safety_score'],
                'calories': round(food_row.get('Calorie', 0) * 200 / 100),  # 200g portion from per 100g data
                'carbs': round(food_row.get('Carbohydrate (g)', 0) * 200 / 100, 1),
                'protein': round(food_row.get('Protein (g)', 0) * 200 / 100, 1),
                'fat': round(food_row.get('Total Fat (g)', 0) * 200 / 100, 1),
                'fiber': round(food_row.get('Dietary Fiber (g)', 0) * 200 / 100, 1),
                'glycemicIndex': food_row.get('GI', 50),
                'glycemicLoad200': gl200,
                'glBadge': _gl_badge(gl200, gl_cutoff),
                'portionSize': "200g (1 serving)",
                'timeOfDay': request.time_of_day,
                'reasons': dynamic_reasons,
                'explanation': food_rec['explanation']
            })
        
        # Final veg-only guard on output, if requested
        if request.meal_preferences and any(p.lower() == 'vegetarian' for p in request.meal_preferences):
            recommendations = [rec for rec in recommendations if is_vegetarian_name(rec.get('name'))]

        return {
            'recommendations': recommendations,
            'user_profile': {
                'bmi': bmi,
                'risk_profile': 'high' if bmi > 30 or request.fasting_sugar > 125 else 'moderate' if bmi > 25 or request.fasting_sugar > 100 else 'low'
            },
            'personalization_factors': [
                f"Age: {request.age} years",
                f"BMI: {bmi:.1f}",
                f"Fasting glucose: {request.fasting_sugar} mg/dL",
                f"Meal time: {request.time_of_day}"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

def generate_intelligent_reasons(food_name: str, food_row: pd.Series, portion_features: Dict, 
                                user_data: Dict, risk_level: str) -> List[str]:
    """Generate intelligent, food-specific reasons for recommendations."""
    reasons = []
    
    # Get nutritional data
    calories = food_row.get('Calorie', 0)
    carbs = food_row.get('Carbohydrate (g)', 0)
    protein = food_row.get('Protein (g)', 0)
    fiber = food_row.get('Dietary Fiber (g)', 0)
    gi = food_row.get('GI', 50)
    fat = food_row.get('Total Fat (g)', 0)
    
    food_lower = food_name.lower()
    
    # Food category specific reasons
    if any(veg in food_lower for veg in ['vegetable', 'sabzi', 'bhindi', 'spinach', 'methi', 'cauliflower']):
        reasons.append(f"Rich in fiber ({fiber:.1f}g) - helps slow glucose absorption")
        if gi < 55:
            reasons.append(f"Low glycemic index ({gi}) prevents blood sugar spikes")
            
    elif any(dal in food_lower for dal in ['dal', 'lentil', 'arhar', 'moong', 'chana']):
        reasons.append(f"High protein ({protein:.1f}g) promotes satiety and stable blood sugar")
        reasons.append("Recommended by diabetologists - 1-2 servings daily")
        
    elif any(grain in food_lower for grain in ['rice', 'roti', 'wheat', 'bread']):
        if gi > 70:
            reasons.append(f"High GI ({gi}) - recommend pairing with vegetables and protein")
        else:
            reasons.append(f"Moderate GI ({gi}) - good carbohydrate choice when portion-controlled")
            
    elif any(fruit in food_lower for fruit in ['apple', 'orange', 'banana', 'fruit']):
        reasons.append(f"Natural fruit sugars with fiber ({fiber:.1f}g) for better glycemic control")
        
    # BMI-specific reasons
    user_bmi = user_data.get('bmi', 25)
    if user_bmi > 25 and calories < 100:
        reasons.append(f"Low calorie ({calories:.0f} kcal) - supports weight management")
    elif user_bmi > 25 and calories > 200:
        reasons.append(f"Higher calorie content - consider smaller portions for weight goals")
        
    # Blood sugar specific reasons
    fasting_sugar = user_data.get('fasting_sugar', 100)
    if fasting_sugar > 125:  # High fasting glucose
        if carbs < 15:
            reasons.append("Low carbohydrate content ideal for glucose control")
        elif carbs > 30:
            reasons.append("Higher carbs - monitor blood sugar 2 hours post-meal")
            
    # Age-specific recommendations
    age = user_data.get('age', 35)
    if age > 50 and protein > 8:
        reasons.append(f"High protein ({protein:.1f}g) supports muscle health in mature adults")
        
    # Risk level specific reasons
    if risk_level == 'safe':
        reasons.append("Multiple safety factors align with your health profile")
    elif risk_level == 'caution':
        reasons.append("Acceptable with portion control and monitoring")
        
    # Time of day reasons
    time_of_day = user_data.get('time_of_day', 'Lunch')
    if time_of_day == 'Breakfast' and carbs > 20:
        reasons.append("Good morning carbs provide sustained energy")
    elif time_of_day == 'Dinner' and carbs < 20:
        reasons.append("Light carbs ideal for evening meal")
        
    # Default fallback
    if not reasons:
        reasons.append("Nutritionally balanced option for diabetic diet")
        
    return reasons[:3]  # Limit to 3 most relevant reasons

# New personalized recommendation request model
class TrulyPersonalizedRequest(BaseModel):
    user_id: int
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    fasting_sugar: float
    post_meal_sugar: int
    diabetes_type: str
    time_of_day: str = "Lunch"
    meal_preferences: Optional[List[str]] = None
    count: int = 6

@app.post("/truly-personalized-recommendations")
async def get_truly_personalized_recommendations(request: TrulyPersonalizedRequest):
    """
    Generate truly personalized meal recommendations using individual user ML models
    trained on User_Logs_Dataset.csv - this replaces generic reasons with user-specific insights.
    """
    global personalized_recommender
    
    if personalized_recommender is None:
        # Fallback to general recommendations if personalized not available
        return await get_general_ml_recommendations(request)
    
    try:
        # Get available foods from the main dataset
        if meal_safety_predictor is None or not hasattr(meal_safety_predictor, 'food_df'):
            raise HTTPException(status_code=503, detail="Food dataset not loaded")
        
        all_foods = list(meal_safety_predictor.food_df.index)
        
        # Filter foods based on time of day
        time_filters = {
            'Breakfast': ['idli', 'dosa', 'poha', 'upma', 'oats', 'daliya', 'paratha', 'milk', 'bread'],
            'Lunch': ['dal', 'rice', 'roti', 'sabzi', 'curry', 'pulao', 'khichdi', 'vegetable'],
            'Dinner': ['soup', 'dal', 'roti', 'sabzi', 'curry', 'vegetable', 'salad'],
            'Snack': ['fruit', 'nuts', 'tea', 'milk', 'sprouts', 'chaat', 'biscuit']
        }
        
        relevant_keywords = time_filters.get(request.time_of_day, time_filters['Lunch'])
        candidate_foods = []
        
        for food in all_foods:
            food_lower = food.lower()
            if any(keyword in food_lower for keyword in relevant_keywords):
                candidate_foods.append(food)

        # Apply vegetarian preference if requested
        if request.meal_preferences and any(p.lower() == 'vegetarian' for p in request.meal_preferences):
            candidate_foods = [f for f in candidate_foods if is_vegetarian_name(f)]
        
        # If no specific matches, use broader selection
        if len(candidate_foods) < request.count:
            # Keep time-of-day alignment; don't drop veg filter once chosen
            candidate_foods = [f for f in all_foods if any(k in f.lower() for k in relevant_keywords)][:50]
        
        # Prepare user features for personalized prediction
        user_features = {
            'Age': request.age,
            'Weight': request.weight_kg,
            'Height': request.height_cm,
            'BMI': request.weight_kg / ((request.height_cm / 100) ** 2),
            'Gender_encoded': 1 if request.gender.lower() == 'male' else 0,
            'Diabetes_Type_encoded': 0 if request.diabetes_type == 'Type1' else 1,
            'Meal_Time_encoded': {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Snack': 3}.get(request.time_of_day, 1)
        }
        
        # Get personalized predictions for each food
        gl_cutoff = _gl_universal_cutoff()
        food_recommendations = []
        for food in candidate_foods[:30]:  # Limit for performance
            try:
                # Get personalized blood sugar prediction
                predicted_bs = personalized_recommender.predict_blood_sugar(
                    request.user_id, food, user_features
                )
                
                # Get personalized recommendation reason
                personalized_reason = personalized_recommender.get_personalized_recommendation_reason(
                    request.user_id, food, predicted_bs
                )
                
                # Determine risk thresholds by diabetes type (conservative)
                dtype = (request.diabetes_type or 'Type2').lower()
                if dtype == 'gestational':
                    safe_thr, caution_thr = 120, 140
                elif dtype == 'prediabetes':
                    safe_thr, caution_thr = 130, 160
                elif dtype == 'type1':
                    safe_thr, caution_thr = 140, 180
                else:  # type2 or others
                    safe_thr, caution_thr = 140, 170

                # Determine risk level based on predicted blood sugar
                if predicted_bs <= safe_thr:
                    risk_level = "safe"
                    safety_score = 1.0
                elif predicted_bs <= caution_thr:
                    risk_level = "caution"
                    safety_score = 0.01
                else:
                    risk_level = "unsafe"
                    safety_score = 0.0
                
                # Get nutritional info
                food_row = meal_safety_predictor.food_df.loc[food]
                # Strict GL threshold by diabetes type: skip when GL for 200g exceeds cutoff
                gl200 = _compute_gl_for_standard_portion(food_row, 200.0)
                if gl200 is None or gl200 >= gl_cutoff:
                    continue
                # Tailoring multipliers
                gi = food_row.get('GI', 50) or 50
                calories200 = round(food_row.get('Calorie', 0) * 200 / 100)
                carbs200 = round(food_row.get('Carbohydrate (g)', 0) * 200 / 100, 1)
                fiber200 = round(food_row.get('Dietary Fiber (g)', 0) * 200 / 100, 1)
                prot200 = round(food_row.get('Protein (g)', 0) * 200 / 100, 1)

                # GI preference by diabetes type
                gi_mult = 1.0
                if dtype in ['type2', 'gestational', 'prediabetes']:
                    if gi <= 55:
                        gi_mult *= 1.12
                    elif gi >= 70:
                        gi_mult *= 0.85

                # BMI-guided calorie moderation
                bmi_val = request.weight_kg / ((request.height_cm / 100) ** 2) if request.height_cm and request.weight_kg else 24.0
                bmi_mult = 1.0
                if bmi_val >= 30:
                    if calories200 > 300: bmi_mult *= 0.8
                    if calories200 < 180: bmi_mult *= 1.05
                elif bmi_val >= 25:
                    if calories200 > 280: bmi_mult *= 0.9
                    if calories200 < 200: bmi_mult *= 1.03

                # Macro balance preference: more fiber/protein is better
                macro_mult = 1.0
                if fiber200 >= 5: macro_mult *= 1.05
                if prot200 >= 15: macro_mult *= 1.05
                if carbs200 >= 60: macro_mult *= 0.9

                safety_score = safety_score * gi_mult * bmi_mult * macro_mult
                
                food_recommendations.append({
                    'name': food,
                    'predicted_blood_sugar': round(predicted_bs, 1),
                    'risk_level': risk_level,
                    'safety_score': safety_score,
                    'personalized_reason': personalized_reason,
                    'calories': calories200,
                    'carbs': carbs200,
                    'protein': prot200,
                    'fat': round(food_row.get('Total Fat (g)', 0) * 200 / 100, 1),
                    'fiber': fiber200,
                    'glycemicIndex': gi,
                    'glycemicLoad200': gl200,
                    'glBadge': _gl_badge(gl200, gl_cutoff),
                    'portionSize': "200g (1 serving)",
                    'timeOfDay': request.time_of_day
                })
                
            except Exception as e:
                # Skip foods that cause errors
                continue

        # Strict policy: return ONLY low-risk (safe) items
        safe_items = [r for r in food_recommendations if r['risk_level'] == 'safe']
        safe_items.sort(key=lambda x: x['safety_score'], reverse=True)
        top_recommendations = safe_items[:request.count]

        # Firestore analytics logging (non-blocking)
        try:
            if FIREBASE_AVAILABLE and firebase_initialized and firestore_db:
                # Determine model used
                model_used = 'general'
                try:
                    if personalized_recommender and hasattr(personalized_recommender, 'user_models') and request.user_id in personalized_recommender.user_models:
                        model_used = 'personal'
                    else:
                        dtype = (request.diabetes_type or '').strip()
                        # Match cohort models case-insensitively
                        if personalized_recommender and hasattr(personalized_recommender, 'general_models_by_diabetes'):
                            keys = list(getattr(personalized_recommender, 'general_models_by_diabetes', {}).keys())
                            if any(k.lower() == dtype.lower() for k in keys):
                                model_used = 'cohort'
                except Exception:
                    pass
                for rec in top_recommendations:
                    try:
                        firestore_db.collection('recommendation_analytics').add({
                            'user_id': request.user_id,
                            'diabetes_type': request.diabetes_type,
                            'model_used': model_used,
                            'food_name': rec.get('name'),
                            'predicted_blood_sugar': rec.get('predicted_blood_sugar'),
                            'risk_level': rec.get('risk_level'),
                            'safety_score': rec.get('safety_score'),
                            'time_of_day': request.time_of_day,
                            'createdAt': firestore.SERVER_TIMESTAMP
                        })
                    except Exception:
                        continue
        except Exception:
            # Never block recommendations on analytics issues
            pass
        
        # Get user's personal insights
        personal_insights = personalized_recommender.get_personal_insights(request.user_id)
        
        # Get user model status
        model_status = personalized_recommender.get_user_model_status(request.user_id)
        
        # Final veg-only guard on output, if requested
        if request.meal_preferences and any(p.lower() == 'vegetarian' for p in request.meal_preferences):
            top_recommendations = [rec for rec in top_recommendations if is_vegetarian_name(rec.get('name'))]

        return {
            'recommendations': top_recommendations,
            'personalization': {
                'user_id': request.user_id,
                'has_personal_model': model_status['has_personal_model'],
                'meal_count': model_status['meal_count'],
                'model_score': model_status['model_score'],
                'personal_insights': personal_insights,
                'personalization_note': (
                    "Recommendations based on your personal meal history and glycemic responses" 
                    if model_status['has_personal_model'] 
                    else "General ML recommendations - log more meals to get personalized insights"
                )
            },
            'user_profile': {
                'bmi': user_features['BMI'],
                'diabetes_type': request.diabetes_type,
                'personalized': model_status['has_personal_model']
            }
        }
        
    except Exception as e:
        # Fallback to general recommendations on error
        print(f"⚠️ Personalized recommendation error: {e}")
        return await get_general_ml_recommendations(request)

async def get_general_ml_recommendations(request):
    """Fallback to general ML recommendations when personalized not available"""
    # Convert to PersonalizedRecommendationRequest format
    general_request = PersonalizedRecommendationRequest(
        age=request.age,
        gender=request.gender,
        weight_kg=request.weight_kg,
        height_cm=request.height_cm,
        fasting_sugar=request.fasting_sugar,
        post_meal_sugar=request.post_meal_sugar,
        diabetes_type=request.diabetes_type,
        time_of_day=request.time_of_day,
        count=request.count
    )
    
    # Get general recommendations and add note about personalization
    general_result = await get_personalized_recommendations(general_request)
    
    # Add personalization status
    general_result['personalization'] = {
        'user_id': request.user_id,
        'has_personal_model': False,
        'meal_count': 0,
        'model_score': None,
        'personal_insights': "Log more meals to unlock personalized recommendations!",
        'personalization_note': "General ML recommendations - personalized model not available"
    }
    
    # Update reasons to indicate they're general
    for rec in general_result['recommendations']:
        if 'reasons' in rec:
            rec['reasons'] = [f"Traditional healthy option: {reason}" for reason in rec['reasons']]
    
    return general_result

if __name__ == "__main__":
    import uvicorn, os
    # Use dynamic port if provided (e.g., Render sets $PORT). Default to 8000 for local dev.
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
