import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

FACE_SIZE = (160, 160)

def detect_biggest_face_rgb(rgb):
    dets = detector.detect_faces(rgb)
    if not dets:
        return None
    det = max(dets, key=lambda d: d['box'][2]*d['box'][3])
    x, y, w, h = det['box']
    x, y = max(0, x), max(0, y)
    x2, y2 = min(rgb.shape[1], x + w), min(rgb.shape[0], y + h)
    return (x, y, x2, y2)

def crop_align(rgb, box):
    if box is None: return None
    x1, y1, x2, y2 = box
    face = rgb[y1:y2, x1:x2]
    if face.size == 0: return None
    return cv2.resize(face, FACE_SIZE)

def embed_face(face_rgb):
    arr = np.asarray(face_rgb, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    emb = embedder.embeddings(arr)[0].astype(np.float32)
    return emb
