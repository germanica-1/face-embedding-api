import numpy as np
import cv2
import base64
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Columns for embeddings
EMBEDDING_COLUMNS = [
    "face_embedding_front",
    "face_embedding_low_angle",
    "face_embedding_left",
    "face_embedding_right"
]

# Columns for images 
IMAGE_URL_COLUMNS = [
    "face_image_front_url",
    "face_image_low_angle_url",
    "face_image_left_url",
    "face_image_right_url"
]


def fetch_students_without_embeddings():
    """
    Fetch students that have at least one missing embedding.
    Returns a list of student dicts.
    """
    try:
        res = supabase.table("students").select("*").execute()
        if not res.data:
            print("[INFO] No students found in the database.")
            return []

        # Filter students with missing embeddings
        students = [
            s for s in res.data
            if any(not s.get(col) for col in EMBEDDING_COLUMNS)
        ]
        return students

    except Exception as e:
        print("[ERROR] Fetching students failed:", e)
        return []


def base64_to_rgb(base64_str):
    """
    Convert a base64-encoded image string to an RGB numpy array.
    Returns None if decoding fails.
    """
    if not base64_str:
        return None
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]

    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("[ERROR] Failed to decode base64 image:", e)
    return None

def clear_base64_images(student_name):
    """
    After embeddings are computed, clear base64 columns to reduce DB size.
    """
    # Replace with your actual Base64 columns in DB
    BASE64_COLUMNS = [
        "face_image_front_url",
        "face_image_low_angle_url",
        "face_image_left_url",
        "face_image_right_url"
    ]

    clear_data = {col: None for col in BASE64_COLUMNS}

    try:
        res = supabase.table("students").update(clear_data).eq("full_name", student_name).execute()
        if res.data and len(res.data) > 0:
            print(f"[INFO] Cleared base64 images for {student_name}")
        else:
            print(f"[WARN] No rows updated when clearing base64 for {student_name}")
    except Exception as e:
        print(f"[ERROR] Failed to clear base64 images for {student_name}:", e)


def update_student_embeddings(student_name, embeddings):
    if not embeddings:
        print(f"[WARN] No embeddings to update for {student_name}")
        return

    data = {}
    for idx, col in enumerate(EMBEDDING_COLUMNS):
        data[col] = embeddings[idx].tolist() if len(embeddings) > idx else None

    try:
        res = supabase.table("students").update(data).eq("full_name", student_name).execute()
        if res.data and len(res.data) > 0:
            print(f"[INFO] Updated embeddings for {student_name}")
    
            clear_base64_images(student_name)

        else:
            print(f"[WARN] No rows updated for {student_name}. Check spelling or column.")
    except Exception as e:
        print(f"[ERROR] Updating embeddings for {student_name} failed:", e)
