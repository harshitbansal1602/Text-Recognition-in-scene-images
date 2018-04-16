# Text-Recognition-in-scene-images
[p]: #project
**[METHODOLOGY][mt] | [INSTALL][i] | [USAGE][u] | [DATA][d]**

Text recognition in scene images using Stroke Width Transform as a text extractor method and a small deep learning model for classification.

## Methodology
[mt]: #methodology 'Methodology guide'

Retrieving texts in both indoor and outdoor environments provides contextual clues for a wide variety of vision tasks.
Moreover, it has been shown that the performance of image retrieval algorithms depends critically on the performance of their text detection modules.
Here I have used two most popular methods to extract text from images namely 'Stroke Width Transform(SWT)' and 'Maximally Stable Extremal Regions (MSER)'.

#### Stroke Width Transform(SWT)
For SWT following [paper](http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf) is refered, also for detailed explanation on swt.py please refer 
[here](https://github.com/mypetyak/StrokeWidthTransform). Note: Currently only one type of text can be recognized at a time by this method i.e. either light text on dark background or vice versa.

```
To detect light text on dark background use the flag 'light_on_dark = True'

In recogize.py:
....
else:
	#Getting regions of image with similar stroke widths
	light_on_dark = False #change here.
	swt_scrubber = swt.SWTScrubber()
	text_boxes = swt_scrubber.scrub(image_path,light_on_dark)
....

```

#### Maximally Stable Extremal Regions (MSER)
For detection of mser's in-built function in opencv is used.

```
To switch between the methods use the flag 'use_mser = True'(Default: False) in recognize.py:

In analysis.py:
....
img = cv2.imread(image_path)
use_mser = False # False: To use SWT, True: To use MSER
....
```

## GETTING STARTED
[gt]: #getting-started 'Getting started guide'

### Install
[i]: #install 'Installation guide'
This project requires **Python** and the following Python libraries installed:
- [SciPy](https://www.scipy.org/install.html)
- [NumPy](https://www.scipy.org/install.html)
- [Pandas](https://www.scipy.org/install.html)
- [scikit-learn](http://scikit-learn.org/stable/)
- [OpenCV](https://docs.opencv.org/trunk/d2/de6/tutorial_py_setup_in_ubuntu)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/install/)

### USAGE
[u]: #usage 'Product usage'
#### Training model

A pre-trained model is included in the /model directory. To train your model, you can use the train.py as a template.
Note: Input shape is set to (32 X 32 X 1) to change it same changes must be made in the 'get_image' function in predict.py: 

```
def get_image(box,img):
    ....
    #To change number of input channels to 3(Default: 1) comment-out this section.
    if image.shape[-1] > 1:
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ....
```
#### Recognize Text

Once the model is trained, you can start recognizing text in natural images. Just give path to the trained model and image path in recognize.py:
```
....

#Loading trained model 
model = load_model('model/model.h5') #give trained model path here.

#Reading image
image_path = 'images/dPICT0032.jpg' #give image path here.

....
```
All the predicted words are printed at the end, and bounding rectangles are made around each.
Note: A feature of spell check is also included in spell_check.py. 

## Data
[d]: #data 'Info about data'
Dataset used in training the model used is provided in /model directory which is taken from [here](https://cs.stanford.edu/people/twangcat/#research). 
For other datasets or resources, have a look at this great [repo](https://github.com/chongyangtao/Awesome-Scene-Text-Recognition).

