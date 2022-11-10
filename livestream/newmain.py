import sys

sys.path.insert(0,'.')
import cv2
import numpy as np
from modules import face_describer_server, face_track_server, camera_server, face_db
from configs import configs

'''
The demo app utilize all servers in model folder with simple business scenario/logics:
I have a camera product and I need to use it to find all visitors in my store who came here before.

Main logics is in the process function, where you can further customize.
'''


class LiveStream(camera_server.CameraServer):

    def __init__(self, *args, **kwargs):
        super(LiveStream, self).__init__(*args, **kwargs)
        self.face_tracker = face_track_server.FaceTrackServer(detector_backend='opencv')
        self.face_describer = face_describer_server.FDServer(model_name=configs.model_name)
        self.face_db = face_db.Model()

    def processs(self, frame):
        # Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Locations)
        faces_loc = self.face_tracker.process(frame)

        print("face location: ", faces_loc)
        # Step2. For each face, get the cropped face area, feeding it to face describer
        # to get 512-D Feature Embedding
        face_descriptions = []
        names = []
        num_faces = len(faces_loc)
        if num_faces == 0:
            #only display the frame in the feed no need to do anythine else
            self.viz_faces([],[],frame)
            return

        for face_loc in faces_loc:
            x, y, w, h = face_loc
            face = frame[y:y + h, x:x + w]
            face_description = self.face_describer.inference(face)
            face_descriptions.append(face_description)
            similar_face_name = self.face_db.who_is_this_face(face_description)
            names.append(similar_face_name)
        
        #Step3. Visualize all the faces in the frame
        self.viz_faces(faces_loc, names, frame)
        print('[Live Streaming] -----------------------------------------------------------')

    def viz_faces(self, faces_loc, names, frame):
        for i in range(len(names)):
            if faces_loc[i] == None:
                continue
            x = int(faces_loc[i][0])
            y = int(faces_loc[i][1])
            width = int(faces_loc[i][2])
            height = int(faces_loc[i][3])
            #size of fill rectangle is to made in proportion with the bbox
            fontsize = height/200
            background = (0, 255, 0)
            if names[i]=="unknown":
                background=(0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+width, y+height), background, 2)
            # Draw a filled rectangle below bounding box for writing name
            cv2.rectangle(frame, (x, y+height), (x+width, int(y+1.1*height)), background, cv2.FILLED)
            # Write name
            cv2.putText(frame,names[i],(x,int(y+1.1*height)),cv2.FONT_HERSHEY_SIMPLEX, fontsize,(255, 255, 255), 1)
        cv2.imshow('faces', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    livestream = LiveStream(camera_address=0)
    livestream.run()

