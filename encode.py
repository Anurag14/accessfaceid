import os
import cv2
import warnings
import face_alignment
from training.vggface import vggface
from training.utils import preprocess_image,execute_alignment
import numpy as np
from tqdm import tqdm
from faced import FaceDetector
from keras.models import Model
def process_and_encode(dataset):
    print("[LOG] Collecting images ...")
    images = []
    for direc, _, files in tqdm(os.walk(dataset)):
        for file in files:
            if file.endswith("jpg"):
                images.append(os.path.join(direc,file))
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    print("[LOG] Encoding faces ...")
    model=vggface()
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    face_detector = FaceDetector()
    face_alignment_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False)
    for image_path in tqdm(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_detector.predict(image)
        if(boxes==[]):
            warnings.warn('system could not detect face in this image %s'%(image_path))
            continue
        (x,y,w,h,prob)=boxes[0]
        #TODO align faces
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        part_image = image[y1:y2,x1:x2]
        landmarks=face_alignment_predictor.get_landmarks(part_image)
        if(landmarks!=[] and landmarks!=None):
            part_image=execute_alignment(part_image,landmarks)
        part_image=preprocess_image(part_image)
        encoding = vgg_face_descriptor.predict(part_image)[0,:]
        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]
        if len(encoding) > 0 : 
            known_encodings.append(encoding)
            known_names.append(name)
    np.savez('data/encodings/encoding.npz',encodings=known_encodings,names=known_names)
    return 

if __name__ == "__main__":
   process_and_encode('data/faces') 
