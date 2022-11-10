import tqdm 
import os
import time 
from deepface import DeepFace
import numpy as np
def process_and_encode(db_path):
	print("[LOG] Collecting images ...")
	images = []
	#check passed db folder exists
	if os.path.isdir(db_path) == True:
		for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
			for file in f:
				if ('.jpg' in file):
					exact_path = r + "/" + file
					images.append(exact_path)

	if len(images) == 0:
		print("WARNING: There is no image in this path ("+db_path+")")
		print("Face recognition will not be performed.")

	tic = time.time()
	#-----------------------
	print('Finding embeddings...')

	

	embeddings = []
	names = []
	#for employee in employees:
	for image_path in images:
		# TODO: Potential threat to remove enforce detection. 
		embedding = DeepFace.represent(image_path, enforce_detection=False)
		embedding = embedding / np.linalg.norm(embedding)
		embeddings.append(embedding)
		names.append(image_path.split('/')[-2])

	toc = time.time()

	print("Embeddings found for given data set in ", toc-tic," seconds")
	np.savez('data/encodings/encoding.npz',encondings=embeddings,names=names)
	return 

if __name__ == "__main__":
	process_and_encode('data/faces') 