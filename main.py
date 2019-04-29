import cv2,os
import numpy as np
from faced import FaceDetector
from faced.utils import annotate_image
from training.vggface import vggface
from training.utils import preprocess_image,findCosineSimilarity
from keras.models import Model
import tensorflow as tf
def prepare_database():
    print("[LOG] Loading Encoded faces ...")
    file = np.load('data/encodings/encoding.npz')
    known_face_encodings=file["encodings"]
    known_face_names = file["names"]
    database={"names":known_face_names,"encodings":known_face_encodings}
    return database



frame_interval=2
def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    model = vggface()
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    face_detector = FaceDetector()
    c=0
    while vc.isOpened():
        _, frame = vc.read()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        timeF = frame_interval
        if(c==0):
            bboxes = face_detector.predict(rgb_img)
            frame = process_frame(bboxes, frame, vgg_face_descriptor)   
            ann_img = annotate_image(frame, bboxes)
        c=(c+1)%timeF
        key = cv2.waitKey(100)
        cv2.imshow("preview", ann_img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(bboxes, frame, vgg_face_descriptor):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    
    for (x, y, w, h, prob) in bboxes:
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        identity = find_identity(frame, x1, y1, x2, y2,vgg_face_descriptor)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, identity, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

    return frame

def find_identity(frame, x1, y1, x2, y2,vgg_face_descriptor):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    #TODO FACE ALIGNMENT HERE
    return who_is_it(part_image, database,vgg_face_descriptor)


def who_is_it(image, database,vgg_face_descriptor,epsilon=0.4):
    """
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = vgg_face_descriptor.predict(preprocess_image(image))[0,:]
    identity = "Unknown"
    # Loop over the database dictionary's names and encodings.
    i=0
    for db_enc in database["encodings"]:
        cosine_similarity = findCosineSimilarity(encoding, db_enc)
        if cosine_similarity < epsilon:
            identity = database["names"][i]
            break
        i+=1
    return identity

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)
    


