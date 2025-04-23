from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import tempfile

app = FastAPI()

class SignatureRequest(BaseModel):
    reference_base64: str
    candidate_base64: str

def decode_base64_image(base64_str: str) -> np.ndarray:
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    return img

def compare_signatures(img1: np.ndarray, img2: np.ndarray) -> float:
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_count = len(matches)
    keypoint_avg = (len(kp1) + len(kp2)) / 2
    similarity = (match_count / keypoint_avg) * 100 if keypoint_avg else 0.0
    return round(min(similarity, 100.0), 2)

@app.post("/compare-signatures-base64")
def compare_signatures_base64(data: SignatureRequest):
    img1 = decode_base64_image(data.reference_base64)
    img2 = decode_base64_image(data.candidate_base64)
    similarity = compare_signatures(img1, img2)
    return {"similarity_percent": similarity}
