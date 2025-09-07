from flask import Flask, jsonify
from processing.face_processing import detect_biggest_face_rgb, crop_align, embed_face
from processing.supabase_helper import (
    fetch_students_without_embeddings,
    update_student_embeddings,
    base64_to_rgb
)
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app) 

@app.route("/api/process-embeddings", methods=["POST"])


def process_embeddings():
    students = fetch_students_without_embeddings()
    if not students:
        return jsonify({"status": "success", "processed": 0, "message": "No students need embeddings"})

    processed_count = 0
    debug_data = {}  # store embeddings for JSON file

    for student in students:
        full_name = student.get("full_name") or student.get("name")
        urls = [
            student.get("face_image_front_url"),
            student.get("face_image_low_angle_url"),
            student.get("face_image_left_url"),
            student.get("face_image_right_url")
        ]

        embeddings = []
        for url in urls:
            rgb = base64_to_rgb(url)
            if rgb is None:
                continue
            box = detect_biggest_face_rgb(rgb)
            face = crop_align(rgb, box)
            if face is None:
                continue
            emb = embed_face(face)
            embeddings.append(emb)

        if embeddings:
            # Save to JSON for debugging
            debug_data[full_name] = [e.tolist() for e in embeddings]

            # Update Supabase
            update_student_embeddings(full_name, embeddings)
            processed_count += 1

    # Write JSON file
    with open("debug_embeddings.json", "w") as f:
        json.dump(debug_data, f, indent=2)

    return jsonify({"status": "success", "processed": processed_count})


if __name__ == "__main__":
    app.run(debug=True)
