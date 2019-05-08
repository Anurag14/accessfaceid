import cv2
import numpy as np
from configs import configs
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
"""
    Face alignment utils 
"""
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape[0][i], eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p[0], points)
    ys = map(lambda p: p[1], points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]

def execute_alignment(img,preds):
    height, width = img.shape[:2]
    left_eye = extract_left_eye_center(preds)
    right_eye = extract_right_eye_center(preds)

    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
    return rotated
"""
    INSIGHT FACE MODEL utils
"""
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
    return face_dist_value


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=configs.face_similarity_threshold):
    true_list = list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    similar_indx = list(np.where(true_list)[0])
    return similar_indx
