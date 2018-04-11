import scipy.io as sio
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import losses
from keras.layers import Input,Reshape,Conv2D,Flatten,Dense,Dropout,MaxPooling2D,BatchNormalization
import sklearn
from load_data import load_data
from sklearn.metrics import confusion_matrix
import string as sr
import time
##Reading data
images,tags = load_data('syntheticData.mat')
tags = to_categorical(tags, num_classes=62)
tags = np.repeat(tags,2,axis=0)
images,tags = sklearn.utils.shuffle(images,tags)


lower_case_list = np.array(list(sr.ascii_lowercase))
upper_case_list = np.array(list(sr.ascii_uppercase))
digits = np.arange(0,10)
chars = np.concatenate((upper_case_list,lower_case_list, digits.astype(str)))


##Hyperparameters

kernel_conv1 = [3,3]
filter_conv1 = 24
kernel_conv2 = [3,3]
filter_conv2 = 24
kernel_conv3 = [3,3]
filter_conv3 = 32
dense_size1 = 128

num_classes = 62 
batch_size = 128
epochs = 50
reg_str = 0.002
adam = optimizers.Adam(lr=0.01, decay=1e-3)

#Model

x_image = Input(shape=[32,32])
x = Reshape([32,32,1])(x_image)
conv_1 = Conv2D(filters = filter_conv1,kernel_size = kernel_conv1,activation = 'relu')(x)
batch_norm_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(batch_norm_1)
# dropout_1 = Dropout(.2)(pool_1)

conv_2 = Conv2D(filters = filter_conv2,kernel_size = kernel_conv2,activation = 'relu')(pool_1)
batch_norm_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(batch_norm_2)
# dropout_2 = Dropout(.2)(pool_2)

conv_3 = Conv2D(filters = filter_conv3,kernel_size = kernel_conv3,activation = 'relu')(pool_2)
batch_norm_3 = BatchNormalization()(conv_3)
pool_3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(batch_norm_3)
# dropout_3 = Dropout(.2)(pool_3)

flat = Flatten()(pool_3)

dense1 = Dense(dense_size1,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(flat)

output = Dense(num_classes,activation = 'softmax')(dense1)

model = Model(inputs= [x_image],outputs = [output])
model.compile(loss= losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

model.summary()
model.fit(images, tags,batch_size=batch_size,epochs=epochs,verbose=1, validation_split = .1)
model.save('model.h5')