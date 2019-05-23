import numpy as np
from configs import configs
import multiprocessing as mp 
"""
    INSIGHT FACE MODEL utils
"""
def face_distance_single_min(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    index = np.argmin(face_dist_value)
    print(f'which index is minimun: {index} and value is {face_dist_value[index]}')
    name = names[index] if face_dist_value[index] <= tolerance else "unknown"
    return name

def face_distance_single_majority(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    similar_indices = np.argwhere(face_dist_value <= tolerance)
    similar_indices = [index[0] for index in similar_indices]
    name_list = list(names[similar_indices])
    if name_list == []:
        return "unknown"
    name = max(set(name_list),key=name_list.count)
    return name

def face_distance_mp_min(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    index = np.argmin(face_dist_value)
    return face_dist_value[index],n_chunk[index]

def face_distance_mp_majority(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
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
    processes=mp.cpu_count()
    for i in range(processes):
        start=int(i*(length_of_db/processes))
        end = length_of_db if i==processes-1 else int((i+1)*(length_of_db/processes))
        chunks.append((known_face_encodings[start:end],names[start:end]))
    pool = mp.Pool(processes)
    results = pool.starmap(face_distance_mp_min,[(e_chunk,n_chunk,face_query) for (e_chunk,n_chunk) in chunks])
    pool.close()
    if mode=='min':
        return min(results, key = lambda t:t[0])[1]
    else:
        return max(set(results),key=results.count)

def compare_faces(known_face_encodings, names, face_query, mode, tolerance=configs.face_similarity_threshold):
    if mode == 'min':
        name= face_distance_single_min(known_face_encodings, names, face_query)
    else:
        name = face_distance_single_majority(known_face_encodings, names, face_query)
    return name
