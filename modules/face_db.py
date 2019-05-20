'''
A dummy db storaing faces in memory
Feel free to make it fancier like hooking with postgres or whatever
This model here is just for simple demo app under apps
Don't use it for production.
'''
from utils.utils_insightface import compare_faces_pp
import numpy as np


class Model(object):

    
    def add_face(self,face_description,name):
        self.faces_descriptions.append(face_description)
        self.faces_names.append(name)
        f_handle = file(self.filename+'_data.npy','a')
        np.save(f_handle,faces_description)
        f_handle.close()
        f_handle = file(self,filename+'_name.npy','a')
        np.save(f_handle,faces_names)
        f_handle.close()

    def __init__(self,model_name='insightface'):
        print("[LOG] Loading Encoded faces Database ...")
        self.filename='data/encodings/encoding_'+model_name
        self.faces_descriptions=np.load(self.filename+'_data.npy')
        self.faces_names = np.load(self.filename+'_name.npy')
    
    def drop_all(self):
        self.faces_names = []
        self.faces_discriptions = []

    def get_all(self):
        return self.faces_names, self.faces_discriptions

    #This method will be deprecated
    def get_similar_faces(self, face_description):
        print('[Face DB] Looking for similar faces in a DataBase of {} faces...'.format(len(self.faces)))
        if len(self.faces) == 0:
            return []
        # Use items in Python 3*, below is by default for Python 2*
        nameof_similar_faces = np.array(self.faces_names)[similar_face_idx]
        num_similar_faces = len(nameof_similar_faces)
        print('[Face DB] Found {} similar faces in a DataBase of {} faces...'.format(num_similar_faces, len(self.faces_names)))
        return nameof_similar_faces
    
    def who_is_this_face(self,face_description,mode='min'):
        assert mode=='min' or mode=='majority'
        if len(self.faces_names) == 0:
            return "Unknown"
        who_is_this = compare_faces_pp(self.faces_descriptions, self.faces_names, face_description,mode)
        return who_is_this
