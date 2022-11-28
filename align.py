import os
import sys
import traceback
import cv2

import numpy as np

from model import Face_detect_crop




def find_faces_and_transformations(
    image: np.ndarray, 
    face_align_model, 
    face_detection_threshold =  0.4, 
    similarity_threshold = 0.2, 
    face_detection_size = (320, 320),
    crop_size = 224):
    
    try:
        faces, transformations = face_align_model.get(
            image, crop_size)
        return faces, transformations
    except Exception as e:
        print(e)
        print(traceback.format_exc())
