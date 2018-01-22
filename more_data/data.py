import scipy.io as sio
import tensorflow as tf
import numpy as np
import cv2
import hdf5storage


mat_contents = hdf5storage.loadmat('bigrams-train.mat')
print mat_contents
print images
images = np.transpose(images, [2,0,1])
for rec in images:
    cv2.imshow('img',rec)
    cv2.waitKey(0)

