import numpy as np
from configs import configs
from statistics import mode
import multiprocessing as mp 
"""
    INSIGHT FACE MODEL utils
"""
def face_distance(face_encodings, names, face_query):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    index = np.argmin(face_dist_value)
    print(f'which index is minimun: {index} and value is {face_dist_value[index]}')
    return face_dist_value[index],names[index]

def face_distance_min(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):

    if len(e_chunk) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    index=np.argmin(face_dist_value)
    return face_dist_value[index],n_chunk[index]

def face_distance_majority(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    if len(e_chunk) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    true_list = list(face_dist_value <= tolerance)
    name_list = names[np.where(true_list)[0]]
    return name_list
"""
    Compare faces parallel process is the module that implements parallel processing of finding the 
    closest image in the data base below the pre defined threshold for the query
"""
def compare_faces_pp(known_face_encodings, names, face_query, mode, tolerance=configs.face_similarity_threshold):
    chunks=[]
    results=[]
    length_of_db=len(known_face_encodings)
    if length_of_db==0:
        return "unknown"
    processes=mp.cpu_count()
    for i in range(processes):
        start=int(i*(length_of_db/processes))
        end = length_of_db if i==processes-1 else int((i+1)*(length_of_db/processes))
        chunks.append((known_face_encodings[start:end],names[start:end]))
    pool = mp.Pool(processes)
    results = pool.starmap(face_distance_min,[(e_chunk,n_chunk,face_query) for (e_chunk,n_chunk) in chunks])
    pool.close()
    if mode=='min':
        return min(results, key = lambda t:t[0])[1]
    else:
        return mode(results)

def compare_faces(known_face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    if len(known_face_encodings) == 0:
        return "unknown"
    face_value,name= face_distance(known_face_encodings, names, face_query)
    name = name if face_value<= tolerance else "unknown"
    return name
