import numpy as np
import cv2

def remove_duplicate(images):
    image_stack = []

    for rec in images:
        
        append = True
        x,y,w,h = rec[:4]
        rec_a = rec
        
        for i,rec2 in enumerate(image_stack): 
            x2,y2,w2,h2 = rec2[:4]
            if  abs(y - y2) <= (h2+h)/4 and abs(x-x2) <= (w2+w)/4 : # remove duplicate box and extend current box 
                if x2 <= x and x <= x2+w2: 
                    if x+w > x2+w2:
                        #box is close and ahead of a previously added box
                        rec_a[:4] = [x2, (y+y2)//2, x-x2+w, max(h,h2)]

                        del image_stack[i]
                    else:
                        append = False

                elif x <= x2 and x <= x2+w2:
                    if x+w < x2+w2:
                        #box is close and ahead of a previously added box
                        append = True
                        rec_a[:4] = [x, (y+y2)//2, x2-x+w2, max(h,h2)]
                        del image_stack[i]
                    else:
                        append = False
                else :
                   pass

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
        if w > 7*h:
            good_append = False
        if h < 4:
            good_append = False

        if good_append:
            good_images.append(rec)
    
    return good_images

def resize_image(images,scale): # Image resizing is performed here
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
  
