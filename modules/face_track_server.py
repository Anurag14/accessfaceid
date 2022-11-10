import cv2
from deepface.detectors import FaceDetector
'''
This server:
    Input: Camera frame
    Output: JSON of Relative locations for each face, with [(tr_x, tr_y, bl_x, bl_y)]

x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


class FaceTrackServer(object):

    def __init__(self, detector_backend):
        self.face_detector = FaceDetector.build_model(detector_backend)
        self.detector_backend = detector_backend
        print("Detector backend is ", detector_backend)
        self.faces = None 

    
    def reset(self):
        self.faces = None

    def process(self, frame):
        self.reset()
        self.faces = FaceDetector.detect_face(self.face_detector, 
                                            img=frame, 
                                            align=True,
                                            detector_backend=self.detector_backend)

        print('[FaceTracker Server] Found {} faces!'.format(len(self.faces)))
        return [self.faces[1]]
