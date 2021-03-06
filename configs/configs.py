import os
from sys import platform
if platform == "win32":
    BASE_PATH = '/'.join(os.getcwd().split('\\')) 
elif platform == "linux":
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) # Using ubuntu machine may require removing this -1
else:
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) 
face_describer_input_tensor_names = ['img_inputs:0', 'dropout_rate:0']
face_describer_output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
face_describer_device = '/cpu:0'
face_describer_model_fp = '{}/pretrained/insightface.pb'.format(BASE_PATH)
face_describer_tensor_shape = (112, 112)
face_describer_drop_out_rate = 0.3
test_img_fp = '{}/tests/test.jpg'.format(BASE_PATH)

face_similarity_threshold = 700
cores = 'single' # single or multi cores 
metric = 'euclidean' #euclidean or cosine or custom
mode = 'min' #min or majority
