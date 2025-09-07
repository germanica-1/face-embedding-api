import cv2
import numpy as np
import sys
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# ---------------- Config ----------------
FACE_SIZE = (160, 160)
THRESHOLD = 0.35   # tune for your dataset/environment
SIM_METRIC = "cosine"
CAM_INDEX = 1      # change if wrong webcam

# ---------------- Supabase ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_COLUMNS = [
    "face_embedding_front",
    "face_embedding_low_angle",
    "face_embedding_left",
    "face_embedding_right",
]

def load_embeddings_from_supabase():
    """
    Load embeddings from Supabase 'students' table.
    Returns dict: { full_name: [np.array, ...], ... }
    """
    db = {}
    try:
        res = supabase.table("students").select("*").execute()
        if not res.data:
            print("[INFO] No student data found in Supabase.")
            return {}

        for student in res.data:
            name = student.get("full_name")
            if not name:
                continue
            vecs = []
            for col in EMBEDDING_COLUMNS:
                arr = student.get(col)
                if arr:
                    try:
                        vecs.append(np.array(arr, dtype=np.float32))
                    except Exception:
                        pass
            if vecs:
                db[name] = vecs
        print(f"[INFO] Loaded {len(db)} students with embeddings from Supabase.")
    except Exception as e:
        print("[ERROR] Failed to fetch embeddings from Supabase:", e)

    return db

# ---------------- Face + Embedding ----------------
detector = MTCNN()
embedder = FaceNet()

def detect_biggest_face_rgb(rgb):
    dets = detector.detect_faces(rgb)
    if not dets:
        return None
    det = max(dets, key=lambda d: d['box'][2] * d['box'][3])
    x, y, w, h = det['box']
    x, y = max(0, x), max(0, y)
    x2, y2 = min(rgb.shape[1], x + w), min(rgb.shape[0], y + h)
    return (x, y, x2, y2)

def crop_align(rgb, box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, FACE_SIZE)

def embed_face(face_rgb):
    arr = np.asarray(face_rgb, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    emb = embedder.embeddings(arr)[0].astype(np.float32)  # already L2-normalized
    return emb

def distance(a, b, metric="cosine"):
    if metric == "cosine":
        return float(1.0 - np.dot(a, b))  # cosine distance
    elif metric == "l2":
        return float(np.linalg.norm(a - b))
    raise ValueError("metric")

def best_match(emb, db):
    best_id, best_d = None, float("inf")
    for sid, vecs in db.items():
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        d = distance(emb, centroid, SIM_METRIC)
        if d < best_d:
            best_id, best_d = sid, d
    return best_id, best_d

# ---------------- Main ----------------
def main():
    db = load_embeddings_from_supabase()
    if not db:
        print("[ERROR] No embeddings loaded. Exiting.")
        sys.exit(1)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("[INFO] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARN: Frame grab failed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = detect_biggest_face_rgb(rgb)

        if box:
            x1, y1, x2, y2 = box
            face = crop_align(rgb, box)
            if face is not None:
                emb = embed_face(face)
                sid, dist = best_match(emb, db)
                if sid and dist <= THRESHOLD:
                    label = f"{sid} ({dist:.3f})"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition (Supabase)", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
