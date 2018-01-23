import scipy.io as sio
import numpy as np
import keras
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import losses
from keras.layers import Input,Reshape,Conv2D,AveragePooling2D,Flatten,Dense,Dropout,MaxPooling2D
import cv2
import sklearn


def normalize(img):
    std = np.std(img)
    if std == 0:
        std = 1.0

    return (img-np.mean(img))/std


def zca(X):
    sigma = np.cov(X, rowvar=True)  # Correlation matrix
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    epsilon = 0.1  # Whitening constant: prevents division by zero
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return np.dot(ZCAMatrix, X)  # Data whitening


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

##Reading data

mat_contents = sio.loadmat('syntheticData.mat')
images = mat_contents['e'][0,0][0]
images = np.transpose(images, [2,0,1])/255.0

tags = mat_contents['e'][0,0][1][0,:] - 1
images = np.asarray([normalize(image) for image in images])
images = np.repeat(images,2,axis=0)
for i in range(1,images.shape[0],2):
    images[i] = -1*images[i]

tags = keras.utils.to_categorical(tags, num_classes=62)
tags = np.repeat(tags,2,axis=0)
images,tags = sklearn.utils.shuffle(images,tags)

##Hyperparameters
kernel_conv1 = [5,5]
filter_conv1 = 12
kernel_conv2 = [5,5]
filter_conv2 = 12
kernel_conv3 = [3,3]
filter_conv3 = 32
kernel_conv4 = [3,3]
filter_conv4 = 32
dense_size1 = 256
dense_size2 = 124

num_classes = 62 #make this 64 by adding random data like false image /bad images
batch_size = 128
epochs = 500
reg_str = 0.002
adam = optimizers.Adam(lr=0.001,decay = 0.001)
x_image = Input(shape=[32,32])
x = Reshape([32,32,1])(x_image)
conv_1 = Conv2D(filters = filter_conv1,kernel_size = kernel_conv1,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(x)
conv_2 = Conv2D(filters = filter_conv2,kernel_size = kernel_conv2,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv_2)

conv_3 = Conv2D(filters = filter_conv3,kernel_size = kernel_conv3,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(pool_1)
conv_4 = Conv2D(filters = filter_conv4,kernel_size = kernel_conv4,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv_4)


flat = Flatten()(pool_2)
flat_d = Dropout(.3)(flat)

dense1 = Dense(dense_size1,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(flat_d)
dense1_d = Dropout(.3)(dense1)

dense2 = Dense(dense_size2,kernel_regularizer=regularizers.l2(reg_str),activation = 'relu')(dense1_d)
dense2_d = Dropout(.2)(dense2)
output = Dense(num_classes,activation = 'softmax')(dense2_d)

model = Model(inputs= [x_image],outputs = [output])
model.compile(loss= losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

model.summary()
model.fit(images, tags,batch_size=batch_size,epochs=epochs,verbose=1, validation_split = .2)
model.save('model.h5')
