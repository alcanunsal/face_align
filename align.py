import os
import traceback
import cv2

import numpy as np

from model import Face_detect_crop

face_detection_threshold = float(os.environ.get("FACE_DET_THRESHOLD", 0.4))
similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.2))
face_det_size = int(os.environ.get("FACE_DET_SIZE", 320))
face_detection_size = (face_det_size, face_det_size)
face_align_model_path = os.path.join(
    "inference", "models_", "simswap", "insightface_func", "models", "antelope")
crop_size = 224

face_align_model = Face_detect_crop(
    root=face_align_model_path)
face_align_model.prepare(
    ctx_id=0, det_thresh=face_detection_threshold, det_size=face_detection_size)


def find_faces_and_transformations(image: np.ndarray):
    try:
        faces, transformations = face_align_model.get(
            image, crop_size)
        return faces, transformations
    except Exception as e:
        print(e)
        print(traceback.format_exc())


imgs_paths = [pth if pth.endswith(
    '.png') or pth.endswith(
    '.jpg') else None for pth in os.listdir("input_photos")]

for img_path in imgs_paths:
    if img_path is None:
        continue
    img = cv2.imread(img)
    faces, _ = find_faces_and_transformations(img)
    cv2.imwrite(f"aligned/{img_path}", faces[0])
