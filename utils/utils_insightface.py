"""
    INSIGHT FACE MODEL utils for face comparison 
"""
import numpy as np
from configs import configs
import multiprocessing as mp 

#################################################################
    Euclidean metric based methods for comparison of faces

#################################################################

def singlecore_euclidean(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements single core based euclidean metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        name = names[index] if face_dist_value[index] <= tolerance else "unknown"
    elif configs.mode=='majority':
        similar_indices = np.argwhere(face_dist_value <= tolerance)
        similar_indices = [index[0] for index in similar_indices]
        name_list = list(names[similar_indices])
        if name_list == []:
            return "unknown"
        name = max(set(name_list),key=name_list.count)
    return name

def multicore_euclidean(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements multi core based euclidean metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        return face_dist_value[index],n_chunk[index]
    elif configs.mode=='majority':
        face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
        true_list = list(face_dist_value <= tolerance)
        name_list = names[np.where(true_list)[0]]
        return name_list

#################################################################
    Cosine metric based methods for comparison of faces

#################################################################

def singlecore_cosine(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements single core based cosine metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        name = names[index] if face_dist_value[index] <= tolerance else "unknown"
    elif configs.mode=='majority':
        similar_indices = np.argwhere(face_dist_value <= tolerance)
        similar_indices = [index[0] for index in similar_indices]
        name_list = list(names[similar_indices])
        if name_list == []:
            return "unknown"
        name = max(set(name_list),key=name_list.count)
    return name

def multicore_cosine(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements multi core based cosine metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        return face_dist_value[index],n_chunk[index]
    elif configs.mode=='majority':
        face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
        true_list = list(face_dist_value <= tolerance)
        name_list = names[np.where(true_list)[0]]
        return name_list

#################################################################
    Custom metric based methods for comparison of faces

#################################################################

def singlecore_custom(face_encodings, names, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements single core based custom metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(face_encodings - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        name = names[index] if face_dist_value[index] <= tolerance else "unknown"
    elif configs.mode=='majority':
        similar_indices = np.argwhere(face_dist_value <= tolerance)
        similar_indices = [index[0] for index in similar_indices]
        name_list = list(names[similar_indices])
        if name_list == []:
            return "unknown"
        name = max(set(name_list),key=name_list.count)
    return name

def multicore_custom(e_chunk, n_chunk, face_query, tolerance=configs.face_similarity_threshold):
    """
        This function implements multi core based custom metric dependent  
        matching corresponding to minimum threshold
    """
    face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
    if configs.mode=='min':
        index = np.argmin(face_dist_value)
        return face_dist_value[index],n_chunk[index]
    elif configs.mode=='majority':
        face_dist_value = np.linalg.norm(e_chunk - face_query, axis=1)
        true_list = list(face_dist_value <= tolerance)
        name_list = names[np.where(true_list)[0]]
        return name_list
    
#################################################################
    Core based calling methods for comparison of faces

#################################################################    
def multicore(known_face_encodings,names,face_query):
    """
    multicore is impl to compare faces in a parallel process. 
    As a module it implements parallel processing of finding the 
    closest image in the data base below the pre defined threshold for the query
    """
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
        results = pool.starmap(multicore_euclidean,[(e_chunk,n_chunk,face_query) for (e_chunk,n_chunk) in chunks])
    pool.close()
    if mode=='min':
        return min(results, key = lambda t:t[0])[1]
    else:
        return max(set(results),key=results.count)

def singlecore(known_face_encodings, names, face_query):
    """
    single core is impl to compare faces on one core
    """
    tolerance, similarity_metric = configs.face_similarity_threshold, configs.metric
    if similarity_metric == 'euclidean':
        name= singlecore_euclidean(known_face_encodings, names, face_query)
    elif similarity_metric == 'cosine':
        name = singlecore_cosine(known_face_encodings, names, face_query)
    else:
        name = singecore_custom(known_face_encodings, names, face_query)
    return name

#################################################################
    Main method for comparison of faces

#################################################################
def comparefaces(known_face_encodings, names, face_query):
    """
    comparefaces as a procedure calls over the single core or multicore implementation of it basis the 
    metric defined by the user.  
    """
    cores = configs.core
    if cores == 'single':
        return singlecore(known_face_encodings, names, face_query)
    elif cores == 'multi':
        return multicore(known_face_encodings, names, face_query)
