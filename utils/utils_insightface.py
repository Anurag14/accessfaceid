import numpy as np
from configs import configs
import multiprocessing as mp 
"""
    INSIGHT FACE MODEL utils
"""
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    #print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
    return face_dist_value
def face_distance_min(face_enodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.min(face_dist_value)


def compare_faces_pp(known_face_encodings, face_encoding_to_check, tolerance=configs.face_similarity_threshold):
    Processes=[]
    length_of_db=len(known_face_encodings)
    no_of_process=mp.cpu_count()
    for i in range(no_of_process):
        start=i*(length_of_db/no_of_process)
        end = length_of_db if i==no_of_process-1 else (i+1)*(length_of_db/no_of_process)
        process = mp.Process(target=face_distance_min, args=(known_face_encodings[start:end],face_to_compare,))
        process.start()
        Processes.append(process)

def compare_faces(known_face_encodings, face_encodings_to_check, tolerance=configs.face_similarity_threshold):
    true_list = list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    similar_indx = list(np.where(true_list)[0])
    return similar_indx
