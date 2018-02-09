import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import cv2

def show_images(images, cols = 1):
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)

    plt.tight_layout() 
    plt.show()

model = load_model('model/model.h5')
weights = []
name = []
for layer in model.layers:
	temp = layer.get_weights()
	if len(temp):
		weights.append((str(layer.name),temp))

for weight in weights:
	
	mat = weight[1][0].transpose((3,0,1,2))
	print mat.shape
	# show_images(mat,4)