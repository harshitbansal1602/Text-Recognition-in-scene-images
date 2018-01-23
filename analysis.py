import numpy as np
from keras.models import load_model
import cv2
import processing
import predict
import string as sr
model = load_model('model.h5')

lower_case_list = list(sr.ascii_lowercase)
upper_case_list = list(sr.ascii_uppercase)
digits = range(0,10)
chars = upper_case_list + lower_case_list + digits 

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
#     print chars[pred]
#     cv2.imshow('img',temp)
#     cv2.waitKey(0)
print predict_images
string = predict.connect_boxes(predict_images)
boxes = [word[0] for word in string]
vis = img.copy()

print string

for box in boxes:
    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)

cv2.imshow('img',vis)
cv2.waitKey(0)
