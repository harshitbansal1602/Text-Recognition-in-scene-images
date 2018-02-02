import numpy as np
import cv2
import processing
import string as sr

def get_image(box,img):

    x,y,w,h = box
    image = img[y:y+h, x:x+w]
    avg_color = [0,0,0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = processing.normalize(image)
    k = max(w,h)
    
    new_dim = k + (32-k%32) if k%32 else k 
    delta_h = new_dim - h
    delta_w = new_dim - w

    top = delta_h-(delta_h//2)
    left = delta_w-(delta_w//2)
    right = delta_w//2
    bottom = delta_h//2

    # temp = np.ones((h+delta_h,w+delta_w))
    # temp[top:top+h, left:left+w] = image
    # image = temp
    image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value = avg_color)

    image = cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
    image = processing.zca_whitening(image)
    return image

def predict(images,img,model):
    
    predict_output = []
    temp = []
    boxes = []
    for rec in images:
        x,y,w,h = rec
        if w>3*h:
            for n in range(x,x+w-h,1):        
                rec2 = [n,y,h,h]
                image = get_image(rec2,img)
                box = rec2
                boxes.append(box)
                temp.append(image)
        else:
            image = get_image(rec,img)
            box = rec
            boxes.append(box)
            temp.append(image)
    for image in temp: 
        cv2.imshow('img',image)
        cv2.waitKey(0)

    probs = model.predict(np.asarray(temp))
    
    for i,prob in enumerate(probs):
        if np.max(prob) > .00:
            rec = boxes[i] #getting boundary box
            predict_output.append([rec,np.argmax(prob)])
    return predict_output


def connect_boxes(images):
    string = []
    
    lower_case_list = list(sr.ascii_lowercase)
    upper_case_list = list(sr.ascii_uppercase)
    digits = range(0,10)
    chars = upper_case_list + lower_case_list + digits 

    while(images.size):
        pred =images[0]
        x1,y1,w1,h1 = pred[0]
        words = [[x1,y1,w1,h1,pred[1]]]
        images = np.delete(images,0,axis = 0)

        j = 0

        while(j < len(images)):
            pred2 = images[j]
            x2,y2,w2,h2 = pred2[0]
            cx2,cy2 = x2+w2/2, y2+h2/2
            word_added = False
            for rec in words:
                x1,y1,w1,h1,_ = rec
                cx1,cy1 = x1+w1/2, y1+h1/2
                #checking proximity
                if abs(cx1-cx2) <= .5*(w1+w2) and abs(cy1-cy2) <= .4*(h1+h2):
                    words.append([x2,y2,w2,h2,pred2[1]])
                    images = np.delete(images,j,axis = 0)
                    word_added = True 
                    break
            if word_added:
                j = 0
            else:
                j += 1

        words = np.array(words)
        words = words[words[:,0].argsort()]

        letters = [chars[i] for i in words[:,-1]]

        #making containg box
        x_min = np.amin(words[:,0])
        y_min = np.amin(words[:,1])
        x_max,arg_x = np.amax(words[:,0]),np.argmax(words[:,0])
        y_max,arg_y = np.amax(words[:,1]),np.argmax(words[:,1])
        #minimum x and y
        box_x = x_min
        box_y = y_min


        box_w = x_max + words[arg_x,2] - x_min
        box_h = y_max + words[arg_y,3] - y_min
        box = [box_x,box_y,box_w,box_h]
        string.append([box,letters])

    return string
    