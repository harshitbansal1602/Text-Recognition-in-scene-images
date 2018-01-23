import numpy as np
import cv2
import processing
import string as sr

def get_image(box,img):

    x,y,w,h = box
    image = img[y:y+h, x:x+w]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = processing.normalize(image)

    delta_h = 32 - h%32 if h%32 else 0
    delta_w = 32 - w%32 if w%32 else 0

    temp = np.ones((h+delta_h,w+delta_w))
    top = delta_h-(delta_h/2)
    left = delta_w-(delta_w/2)

    temp[top:top+h, left:left+w] = image
    image = cv2.resize(temp, (32,32), interpolation=cv2.INTER_AREA)
    
    return image

def predict(images,img,model):
    
    predict_output = []
    temp = []
    boxes = []
    for rec in images:
        x,y,w,h = rec
        if w>2*h:
            for n in range(x,x+w-h,1):
                rec2 = [n,y,h,h]
                image = get_image(rec2,img)
                box = rec2 
        else:
            image = get_image(rec,img)
            box = rec

        boxes.append(box)
        temp.append(image)
    probs = model.predict(np.asarray(temp))
    
    for i,prob in enumerate(probs):
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
        delete = [0]

        for j in range(1,len(images)):
            
            pred2 = images[j]
            x2,y2,w2,h2 = pred2[0]
            cx2,cy2 = x2+w2//2,y2+h2//2

            for rec in words:
                x1,y1,w1,h1,_ = rec
                cx1,cy1 = x1+w1//2,y1+h1//2
                #checking proximity
                if abs(cx1-cx2) <= 1.15*(w1+w2)/2 and abs(cy1-cy2) <= 1.15*(h1+h2)/2:
                    words.append([x2,y2,w2,h2,pred2[1]])
                    delete.append(j)
                    break

        words = np.array(words)
        words =words[words[:,0].argsort()]
        
        images = np.delete(images,delete,axis = 0)
        letters = [chars[i] for i in words[:,-1]]

        #making containg box

        box_x = words[0,0]
        box_y = words[0,1]
        box_w = words[-1,0] + words[-1,2] - box_x
        box_h = np.amax(words[:,3]).astype(int)
        box = [box_x,box_y,box_w,box_h]
        string.append([box,letters])

    return string
    