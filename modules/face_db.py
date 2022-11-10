'''
A dummy db storaing faces in numpy .npy file memory
Feel free to make it fancier like hooking with postgres or whatever
This model here is just for simple demo app under apps
Don't use it for production.
'''
import numpy as np


class Model(object):

    
    def add_face(self,face_description,name):
        face_description = face_description/np.linalg.norm(face_description) #normalize the 512 D embedding
        self.faces_descriptions = np.append(self.faces_descriptions,[face_description],axis=0)
        self.faces_names = np.append(self.faces_names,name)
        np.savez('data/encodings/encoding.npz',encondings=self.faces_descriptions,names=self.face_names)

    def __init__(self,model_name='ArcFace'):
        print("[LOG] Loading Encoded faces Database ...")
        self.file = np.load('data/encodings/encoding.npz')
        self.faces_descriptions=self.file['encondings']
        self.faces_names = self.file['names']
    
    def drop_all(self):
        self.faces_names = []
        self.faces_discriptions = []

    def get_all(self):
        return self.faces_names, self.faces_discriptions

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def who_is_this_face(self,face_description):
        if len(self.faces_names) == 0 or len(self.faces_descriptions)==0:
            return "unknown"
        similarity = self.faces_descriptions @ face_description
        who_is_this = self.faces_names[np.argmax(self.softmax(similarity))]
        return who_is_this
