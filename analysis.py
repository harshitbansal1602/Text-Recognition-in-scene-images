import numpy as np
from keras.models import load_model
import cv2
import processing
import predict
import string as sr
import spell_check
import swt

#Loading trained model 
model = load_model('model/model.h5')

#Reading image
image_path = 'images/bPICT0016.jpg'
img = cv2.imread(image_path)

#Getting regions of image with similar stroke widths
swt_scrubber = swt.SWTScrubber()
boxes = swt_scrubber.scrub(image_path)

#Removing duplicates and removing patches which are part of another bigger patch i.e. it is a part of bigger letter 
good_patches = processing.remove_duplicate(boxes)
predict_patches =  np.array(predict.predict(good_patches, img, model))
#Scaling patches so they can be connected to other overlapping patches 
predict_patches = np.array(processing.resize_image(predict_patches, 1.3))

lower_case_list = list(sr.ascii_lowercase)
upper_case_list = list(sr.ascii_uppercase)
digits = range(0,10)
chars = upper_case_list + lower_case_list + digits 

string = predict.connect_boxes(predict_patches)
predict_boxes = [word[0] for word in string]
strings = [word[1] for word in string]
vis = img.copy()

#correcting known english words, check spell_check.py for more info
correct = [spell_check.correction(''.join([i.lower() for i in word])) for word in strings] 
words = [''.join([i for i in word]) for word in strings]

#Note letters are convertes to lower case.
print np.transpose(correct)
print np.transpose(words)

for box in good_patches:
    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,250,0),2)

cv2.imshow('img',vis)
cv2.waitKey(0)

vis = img.copy()
for box in predict_boxes:
    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,250,0),2)

cv2.imshow('img',vis)
cv2.waitKey(0)
