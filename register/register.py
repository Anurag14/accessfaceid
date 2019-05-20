import sys

sys.path.insert(0,'.')
import cv2
import numpy as np
from modules import face_track_server, face_describer_server, face_db, camera_server,face_align_server
from configs import configs

'''
The register app utilize all servers in model 
I have a camera product and I need to use it to find all visitors in my store who came here before.
If unmatched I need to register this new face

Process function does majority of heavy lifting and call if you want to, that is where you can further customize.
'''


class Register(camera_server.CameraServer):
    
    def __init__(self, name, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device)
        self.face_db = face_db.Model()
        self.name=name

    def enter_face(self,_face_description):
        self.face_db.add_face(face_description=_face_description, name=self.name)
        return

    def processs(self, frame):
        # Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()

        # Uncomment below to visualize face
        _faces_loc = self.face_tracker.get_faces_loc()
        self._viz_faces(_faces_loc, frame)

        # Step2. For each face, get the cropped face area, feeding it to face describer (insightface) to get 512-D Feature Embedding
        _face_descriptions = []
        _num_faces = len(_faces)
        if _num_faces == 0:
            print("No faces found cant register")
            return
        for _face in _faces:
            _face_resize = cv2.resize(_face, configs.face_describer_tensor_shape)
            _data_feed = [np.expand_dims(_face_resize, axis=0), configs.face_describer_drop_out_rate]
            _face_description = self.face_describer.inference(_data_feed)[0][0]
            _face_descriptions.append(_face_description)

            # Step3. For each face, check whether there are similar faces and if not save it to db.
            # Below naive and verbose implementation is to tutor you how this work
            _similar_face_name = self.face_db.who_is_this_face(_face_description, cores='single')
            if _similar_face_name == "unknown" or len(self.face_db.faces_names) == 0:
                self.enter_face(_face_description)
            
            print('[Live Streaming] -----------------------------------------------------------')

    def _viz_faces(self, faces_loc, frame):
        for _face_loc in faces_loc:
            x1 = int(_face_loc[0] * self.face_tracker.cam_w)
            y1 = int(_face_loc[1] * self.face_tracker.cam_h)
            x2 = int(_face_loc[2] * self.face_tracker.cam_w)
            y2 = int(_face_loc[3] * self.face_tracker.cam_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Register', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    name=input("Who do I need to register, Enter Details")
    register = Register(name,camera_address=0)
    register.run()

