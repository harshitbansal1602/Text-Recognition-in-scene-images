import scipy.io as sio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.layers import Activation,Input,Conv2D,Reshape,AveragePooling2D,Flatten,Dense,Dropout,MaxPooling2D
from keras.initializers import Constant
import keras.backend as K
import glob
import cv2
import processing
import predict

model = load_model('model.h5')


img = cv2.imread('Memes/FB_IMG_1505298786198.jpg')
IMG_HEIGHT,IMAGE_WIDTH = img.shape[0],img.shape[1]
mser = cv2.MSER_create()
mser.setDelta(10)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)
hulls = []
for p in regions:
    x,y,w,h = cv2.boundingRect(p.reshape(-1, 1, 2))
    hulls.append([x,y,w,h])

crop_images = np.array(hulls)

good_images = processing.clean_images(crop_images)
good_images = processing.remove_duplicate(good_images)
predict_images =  np.array(predict.predict(good_images,img,model))
predict_images = np.array(processing.resize_image(predict_images,1.3))

# for image in predict_images:
#     pred = image[1]
#     rec = image[0]
#     temp = img[rec[1]:rec[1]+rec[3], rec[0]:rec[0]+rec[2]]
#     print rec
#     cv2.imshow('img'+str(chars[pred]),temp)
#     cv2.waitKey(0)



#grouping letters togather
string = predict.connect_boxes(predict_images)
boxes = [word[0] for word in string]
vis = img.copy()

print string
for box in boxes:
    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)

cv2.imshow('img',vis)
cv2.waitKey(0)
