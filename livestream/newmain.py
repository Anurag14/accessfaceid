import sys

sys.path.insert(0,'.')
import cv2
import numpy as np
from modules import face_track_server, face_describer_server, face_db, camera_server
from configs import configs

'''
The demo app utilize all servers in model folder with simple business scenario/logics:
I have a camera product and I need to use it to find all visitors in my store who came here before.

Main logics is in the process function, where you can further customize.
'''


class Demo(camera_server.CameraServer):

    def __init__(self, *args, **kwargs):
        super(Demo, self).__init__(*args, **kwargs)
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device)
        self.face_db = face_db.Model()

    def processs(self, frame):
        # Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()

        # Step2. Get all the face locations aka bounding boxes
        _faces_loc = self.face_tracker.get_faces_loc()

        # Step3. For each face, get the cropped face area, feeding it to face describer (insightface) to get 512-D Feature Embedding
        _face_descriptions = []
        _names = []
        _num_faces = len(_faces)
        if _num_faces == 0:
            #only display the frame in the feed no need to do anythine else
            self._viz_faces([],[],frame)
            return
        for _face in _faces:
            _face_resize = cv2.resize(_face, configs.face_describer_tensor_shape)
            _data_feed = [np.expand_dims(_face_resize, axis=0), configs.face_describer_drop_out_rate]
            _face_description = self.face_describer.inference(_data_feed)[0][0]
            _face_descriptions.append(_face_description)

            # Step4. For each face, check whether get its name
            # Below naive and verbose implementation is to tutor you how this work
            _similar_face_name = self.face_db.who_is_this_face(_face_description)
            _names.append(_similar_face_name)
        #Step5. Visualize all the faces in the frame
        self._viz_faces(_faces_loc,_names, frame)
        print('[Demo] -----------------------------------------------------------')

    def _viz_faces(self, faces_loc, names, frame):
        for i in range(len(names)):
            x1 = int(faces_loc[i][0] * self.face_tracker.cam_w)
            y1 = int(faces_loc[i][1] * self.face_tracker.cam_h)
            x2 = int(faces_loc[i][2] * self.face_tracker.cam_w)
            y2 = int(faces_loc[i][3] * self.face_tracker.cam_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw a filled rectangle below bounding box for writing name
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            # Write name
            cv2.putText(frame,names[i],(x1 + 6, y2 - 6),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 1)
        cv2.imshow('faces', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    demo = Demo(camera_address=0)
    demo.run()

