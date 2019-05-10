import numpy as np
from configs import configs
"""
    INSIGHT FACE MODEL utils
"""
def get_embedding(part_image,face_descriptor,dropout_rate=0.5):
    input_data=[np.expand(part_image,axis=0),dropout_rate]
    prediction=face_descriptor.inference(data=input_data)
    return prediction

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
