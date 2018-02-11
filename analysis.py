import numpy as np
from keras.models import load_model
import cv2
import processing
import predict
import string as sr
import spell_check
import swt


def show_rectangles(vis,boxes):
	vis = img.copy()
	for box in boxes:
	    cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,250,0),2)

	cv2.imshow('img',vis)
	cv2.waitKey(0)

#Loading trained model 
model = load_model('model/model.h5')

#Reading image
image_path = 'images/dPICT0031.jpg'
img = cv2.imread(image_path)

use_mser = True
if use_mser:
	#Getting regions of image using mser 
	text_boxes = processing.create_mser_regions(img)
else:
	#Getting regions of image with similar stroke widths
	light_on_dark = False
	swt_scrubber = swt.SWTScrubber()
	text_boxes = swt_scrubber.scrub(image_path,light_on_dark)


#Removing duplicates and removing patches which are part of another bigger patch i.e. it is a part of bigger letter 
good_patches = processing.remove_duplicate(text_boxes)
predict_patches =  np.array(predict.predict(good_patches, img, model))
#Scaling patches so they can be connected to other overlapping patches 
predict_patches = np.array(processing.resize_image(predict_patches, 1.3))

#All locations and letters of connected boxes is stored
string = predict.connect_boxes(predict_patches)

boxes = [word[0] for word in string]
letters = [word[1] for word in string]
vis = img.copy()
print letters
#correcting known english words, check spell_check.py for more info
correct = [spell_check.correction(''.join([i.lower() for i in char])) for char in letters] 
words = [''.join([i for i in char]) for char in letters]


print 'Before spell check:',np.transpose(words)
#Note letters are convertes to lower case.
print 'After spell check:',np.transpose(correct)

show_rectangles(img,good_patches)
# Make rectangle around predicted letters
show_rectangles(img,boxes)

