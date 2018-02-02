import numpy as np
from keras.models import load_model
import cv2
import processing
import predict
import string as sr
import spell_check

model = load_model('model.h5')

lower_case_list = list(sr.ascii_lowercase)
upper_case_list = list(sr.ascii_uppercase)
digits = range(0,10)
chars = upper_case_list + lower_case_list + digits 

image_path = 'images/FB_IMG_1509436608392.jpg'
img = cv2.imread(image_path)

#detect regions in gray scale image
crop_images = processing.create_mser_regions(img)
good_images = processing.clean_images(crop_images)
good_images = processing.remove_duplicate(good_images)
predict_images =  np.array(predict.predict(good_images,img,model))
predict_images = np.array(processing.resize_image(predict_images,1.3)) #enlarging images so that we can connect them

# for image in predict_images:
#     pred = image[1]
#     rec = image[0]
#     temp = img[rec[1]:rec[1]+rec[3], rec[0]:rec[0]+rec[2]]
#     print chars[pred]	
#     cv2.imshow('img',temp)
#     cv2.waitKey(0)

string = predict.connect_boxes(predict_images)
boxes = [word[0] for word in string]
strings = [word[1] for word in string]
vis = img.copy()

correct = [spell_check.correction(''.join([i.lower() for i in word if type(i) != int ])) for word in strings]

print np.transpose(strings)
print np.transpose(correct)

for box in good_images:
    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,250,0),2)

cv2.imshow('img',vis)
cv2.waitKey(0)
