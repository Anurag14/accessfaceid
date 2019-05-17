import sys

sys.path.insert(0,'.')
import os
import cv2
import tables
import face_alignment
import numpy as np
from modules import face_track_server, face_describer_server
from utils.utils_alignment import execute_alignment
from configs import configs
def process_and_encode(dataset,filename):
    print("[LOG] Collecting images ...")
    images = []
    for direc, _, files in os.walk(dataset):
        for file in files:
            if file.endswith("jpg"):
                images.append(os.path.join(direc,file))
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    print("[LOG] Encoding faces ...")
    face_descriptor = face_describer_server.FDServer(
                model_fp=configs.face_describer_model_fp,
                input_tensor_names=configs.face_describer_input_tensor_names,
                output_tensor_names=configs.face_describer_output_tensor_names,
                device=configs.face_describer_device)
    face_detector = face_track_server.FaceTrackServer(down_scale_factor=1)
    face_alignment_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False)
    names=[]
    encodings=[]
    i=0
    for image_path in images:
        print(f'i:{i}')
        i+=1
        image = cv2.imread(image_path)
        face_detector.process(image,called_from_encode=False)
        face = face_detector.get_faces()

        face_descriptions = []
        num_faces = len(face)
        if num_faces == 0:
            print(f'Warning no face in the image for the image path {image_path}\n')
            continue
        face=face[0]
        print("[ALIGNMENT]-------STARTS")
        landmarks=face_alignment_predictor.get_landmarks(face)
        if(landmarks!=None):
            face=execute_alignment(face,landmarks)
        print("[ALIGNMENT]-------ENDS")
        face_resize = cv2.resize(face, configs.face_describer_tensor_shape)
        data_feed = [np.expand_dims(face_resize, axis=0), configs.face_describer_drop_out_rate]
        encoding = face_descriptor.inference(data_feed)[0][0]
        # the person's name is the name of the folder where the image comes from
        names.append(image_path.split(os.path.sep)[-2])
        encodings.append(encoding)
    np.save(filename+'_data.npy',encodings)
    np.save(filename+'_name.npy',names)
    return 

if __name__ == "__main__":
    filename = 'data/encodings/encoding_insightface'
    process_and_encode('data/faces',filename) 
