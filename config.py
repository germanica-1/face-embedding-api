# config.py
from dotenv import load_dotenv
import os

load_dotenv()  # load from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or KEY not set in environment variables!")

# FaceNet & face recognition config
FACE_SIZE = (160, 160)
THRESHOLD = 0.35
SIM_METRIC = "cosine"
