'''
A dummy db storaing faces in memory
Feel free to make it fancier like hooking with postgres or whatever
This model here is just for simple demo app under apps
Don't use it for production.
'''
from utils.utils_insightface import compare_faces
import numpy as np


class Model(object):

    
    def add_face(self,face_description,name):
        self.faces_discriptions.append(face_description)
        self.faces_names.append(name)
        np.savez(self.filename,encodings=self.faces_descriptions,names=self.faces_names)

    def __init__(self,model_name='insightface'):
        print("[LOG] Loading Encoded faces Database ...")
        self.filename='data/encodings/encoding_'+model_name+'.npz'
        file=np.load(self.filename)
        self.faces_descriptions=file["encodings"]
        self.faces_names = file["names"]
    
    def drop_all(self):
        self.faces_names = []
        self.faces_discriptions = []

    def get_all(self):
        return self.faces_names, self.faces_discriptions

    def get_similar_faces(self, face_description):
        print('[Face DB] Looking for similar faces in a DataBase of {} faces...'.format(len(self.faces_names)))
        if len(self.faces_names) == 0:
            return []
        # Use items in Python 3*, below is by default for Python 2*
        similar_face_idx = compare_faces(self.faces_discriptions, face_description)
        nameof_similar_faces = np.array(self.faces_names)[similar_face_idx]
        num_similar_faces = len(nameof_similar_faces)
        print('[Face DB] Found {} similar faces in a DataBase of {} faces...'.format(num_similar_faces, len(self.faces_names)))
        return nameof_similar_faces
