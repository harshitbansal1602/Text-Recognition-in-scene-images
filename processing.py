import numpy as np
import cv2


def create_mser_regions(img):
    
    #setting variation
    mser = cv2.MSER_create(_delta = 0, _min_area = 64, _max_area = 640000, _max_variation = 0.4, _min_diversity = 0.05, _max_evolution = 5000, _area_threshold = 1.01, _min_margin = .03, _edge_blur_size = 5)
    regions, _ = mser.detectRegions(img)
    hulls = []
    for p in regions:
        x,y,w,h = cv2.boundingRect(p.reshape(-1, 1, 2))
        hulls.append([x,y,w,h])

    patch_images = np.array(hulls)
    return patch_images


def remove_duplicate(images):
    image_stack = []

    for rec in images:
        
        append = True
        x,y,w,h = rec[:4]
        rec_a = rec
        

        for i,rec2 in enumerate(image_stack): 
            x2,y2,w2,h2 = rec2[:4]

            if  abs(y-y2) <= .7*(h2+h) and abs(x-x2) <= .4*(w2+w) and abs((x+w)-(x2+w2)) <= .4*(w+w2) and abs((y+h)-(y2+h2)) <= .7*(h+h2):
                if x <= x2 and y <= y2 and x+w >= x2+w2 and y+h > y2+h2 :
                    del image_stack[i]
                else:
                    append = False 

        if append:
            image_stack.append(rec_a)

    return np.array(image_stack)

def normalize(img):
    std = np.std(img)
    if std == 0:
        std = 1.0

    return (img-np.mean(img))/std

def clean_images(images):
    
    good_images = []
    for rec in images:
        good_append = True
        
        x,y,w,h = rec
        #filter bad images i.e. veru high aspect ratio or very small height
        if w > 7*h or h < 4 or w < 5 or w > 45 or h > 45:
            good_append = False

        if good_append:
            good_images.append(rec)
    return good_images

def resize_image(images,scale):
    image_stack = []
    for rec in images:
        x,y,w,h = rec[0]
        if w>10:
            x = max(x - int((scale-1)*w/2),0)
            w = int(scale*w)
        
        else:
            x = x - (10-w)/2     
            w = 10
        y = y - int((scale-1)*h/2)
        h = int(scale*h)

        rec2 = []
        rec2 = [x,y,w,h]
        
        image_stack.append((rec2,rec[1]))
    return image_stack
  
