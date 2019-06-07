import numpy as np
from configs import configs
import multiprocessing as mp 
"""
    INSIGHT FACE MODEL utils
"""
### Euclidean face similarity functions 

def singlecore_min_euclidean(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    index = np.argmin(face_dist_value)
    print(f'which index is minimun: {index} and value is {face_dist_value[index]}')
    name = names[index] if face_dist_value[index] <= tolerance else "unknown"
    return name

def singlecore_majority_euclidean(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    similar_indices = np.argwhere(face_dist_value <= tolerance)
    similar_indices = [index[0] for index in similar_indices]
    name_list = list(names[similar_indices])
    if name_list == []:
        return "unknown"
    name = max(set(name_list),key=name_list.count)
    return name

def multicore_min_euclidean(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    index = np.argmin(face_dist_value)
    return face_dist_value[index],n_chunk[index]

def multicore_majority_euclidean(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    true_list = list(face_dist_value <= tolerance)
    name_list = names[np.where(true_list)[0]]
    return name_list

### Cosine face similarity functions 

"""
    multicore is impl to compare faces in a parallel process. 
    As a module it implements parallel processing of finding the 
    closest image in the data base below the pre defined threshold for the query
"""
def multicore(known_face_encodings,names,face_query):
    mode, tolerance, similarity_metric = configs.mode, configs.face_similarity_threshold, configs.metric
    chunks, results =[], []
    length_of_db=len(known_face_encodings)
    processes=mp.cpu_count()
    for i in range(processes):
        start=int(i*(length_of_db/processes))
        end = length_of_db if i==processes-1 else int((i+1)*(length_of_db/processes))
        chunks.append((known_face_encodings[start:end],names[start:end]))
    pool = mp.Pool(processes)
    if mode =='min':
        results = pool.starmap(multicore_min_euclidean,[(e_chunk,n_chunk,face_query) for (e_chunk,n_chunk) in chunks])
    pool.close()
    if mode=='min':
        return min(results, key = lambda t:t[0])[1]
    else:
        return max(set(results),key=results.count)
"""
    single core is impl to compare faces on one core
"""
def singlecore(known_face_encodings, names, face_query):
    mode, tolerance, similarity_metric = configs.mode, configs.face_similarity_threshold, configs.metric
    if mode == 'min' and similarity_metric == 'euclidean':
        name= singlecore_min_euclidean(known_face_encodings, names, face_query)
    elif mode == 'majority' and similarity_metric == 'euclidean':
        name = singlecore_majority_euclidean(known_face_encodings, names, face_query)
    elif mode == 'min' and similarity_metric == 'cosine':
        name = singlecore_min_cosine(known_face_encodings, names, face_query)
    else:
        name = singecore_min_cosine(known_face_encodings, names, face_query)
    return name
"""
    comparefaces as a procedure calls over the single core or multicore implementation of it basis the 
    metric defined by the user.  
"""
def comparefaces(known_face_encodings, names, face_query):
    cores = configs.core
    if cores == 'single':
        return singlecore(known_face_encodings, names, face_query)
    elif cores == 'multi':
        return multicore(known_face_encodings, names, face_query)
