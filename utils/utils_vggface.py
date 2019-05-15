import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
"""
    VGG FACE MODEL utils
"""
def get_embedding(part_image,face_descriptor,dropout_rate=0.5):
    input_data=[np.expand(part_image,axis=0),dropout_rate]
    prediction=face_descriptor.inference(data=input_data)
    return prediction
def preprocess_image(img):
    img = cv2.resize(img,(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def verifyFace(vgg_face_descriptor,img1, img2,epsilon=0.4):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    #euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    return True if (cosine_similarity < epsilon) else False
