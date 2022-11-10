import os
from sys import platform
if platform == "win32":
    BASE_PATH = '/'.join(os.getcwd().split('\\')) 
elif platform == "linux":
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) # Using ubuntu machine may require removing this -1
else:
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) 

model_name = 'VGG-Face'
cores = 'single' # single or multi cores 
metric = 'euclidean' #euclidean or cosine or custom
mode = 'min' #min or majority
