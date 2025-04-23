from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from typing import Tuple
import shutil
import tempfile

app = FastAPI()

def compare_signatures_orb(img1_path: str, img2_path: str) -> float:
    """Returns a similarity score (0 to 100%) using ORB feature matching."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0.0

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity based on how many good matches exist
    match_count = len(matches)
    keypoint_avg = (len(kp1) + len(kp2)) / 2
    if keypoint_avg == 0:
        return 0.0

    similarity = (match_count / keypoint_avg) * 100
    return round(min(similarity, 100.0), 2)

def save_temp_file(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily and return the file path."""
    suffix = upload_file.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

@app.post("/compare-signatures")
async def compare_signatures(
    reference: UploadFile = File(...),
    candidate: UploadFile = File(...)
):
    ref_path = save_temp_file(reference)
    cand_path = save_temp_file(candidate)

    similarity = compare_signatures_orb(ref_path, cand_path)

    return {"similarity_percent": similarity}
