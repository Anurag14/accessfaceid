from deepface import DeepFace
from deepface.commons import functions

class BaseServer(object):
    
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.model = DeepFace.build_model(model_name)
        print(model_name," is built")
        """
        input_shape = functions.find_input_shape(self.model)
        self.input_shape_x = input_shape[0]
        self.input_shape_y = input_shape[1]

        #tuned thresholds for model and metric pair
        self.threshold = dst.findThreshold(self.model_name, self.distance_metric)
        """

    def inference(self, data):
        print('[Base Server] Running inference...')
        embedding = DeepFace.represent(data, model_name= self.model_name, 
            model = self.model,
            enforce_detection = False)
        return embedding