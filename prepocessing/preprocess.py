from imutils.perspective import four_point_transform
import numpy as np
from imutils import contours
import imutils,time
import cv2
# from check_result import ANSWER_KEY_1,ANSWER_KEY_2,ANSWER_KEY_3,ANSWER_KEY_4,ANSWER_KEY_5,TEST_ID


def read_image(img):
    """Read image from file"""
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged,gray



def separate_contours(image,gray_image):
    """Finding main contours of test"""
    X=0
    Y=0
    # origin_image=[0,0,0,0,0,0]
    contours=[0,0,0,0,0,0]
    cnts=cv2.findContours(image,cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_width=max([cv2.boundingRect(c)[2] for c in cnts])
    max_height=max([cv2.boundingRect(c)[3] for c in cnts])
    if max_height>max_width:
	    max_height=max_height
    else:
	    max_height=sum(sorted([cv2.boundingRect(c)[3] for c in cnts])[-2:])
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        if 450>w>300 and 1050>h>950 or 450>w>300 and 2050>h>1700 or 700>w>500 and 800>h>700:
            warped=gray_image[y:y+h,x:x+w]
            if int(0.04*max_width)<=x<=int(0.1*max_width):
                warped_test=warped[int(0.05*h):y+h,int(0.15*w):X+w]
                contours[1]=warped_test
            elif int(0.2*max_width)<=x<=int(0.3*max_width):
                warped_test=warped[int(0.06*h):y+h,int(0.18*w):X+w]
                contours[2]=warped_test
            elif int(0.4*max_width)<=x<=int(0.5*max_width):
                warped_test=warped[int(0.06*h):y+h,int(0.15*w):X+w]
                contours[3]=warped_test
            elif int(0.6*max_width)<=x<=int(0.7*max_width):
                warped_test=warped[int(0.03*h):y+h,int(0.15*w):X+w]
                contours[4]=warped_test
            elif int(0.8*max_width)<=x<=int(0.9*max_width):
                warped_test=warped[int(0.03*h):y+h,int(0.15*w):X+w]
                contours[5]=warped_test
            elif int(0.15*max_width)<=x<=int(0.19*max_width):
                test_id=warped[int(0.13*h):y+h,int(0.01*w):x+w]
                contours[0]=test_id
    return contours


def separate_answers(contours):
    """Separate all answers"""
    thresholded_image=[0,0,0,0,0,0]
    first_answers=[]
    second_answers=[]
    third_answers=[]
    fourth_answers=[]
    fifth_answers=[]
    test_id=[]
    for ind,warped in enumerate(contours):
        thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 11)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
	            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if ind==0:
                thresholded_image[0]=thresh
                if 100>w >= 25 and 100>h >= 25 and ar >= 0.9 and ar <= 1.2:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    test_id.append(c)
                
            if ind==1:
                thresholded_image[1]=thresh
                if 100>w >= 40 and 100>h >= 40 and ar >= 0.8 and ar <= 1.3:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    first_answers.append(c)
                    
                
            elif ind==2:
                thresholded_image[2]=thresh
                if 100>w >= 40 and 100>h >= 40 and ar >= 0.8 and ar <= 1.3:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    second_answers.append(c)
               
            elif ind==3:
                thresholded_image[3]=thresh
                if 100>w >= 40 and 100>h >= 40 and ar >= 0.9 and ar <= 1.3:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    third_answers.append(c)
               
            elif ind==4:
                thresholded_image[4]=thresh
                if 100>w >= 40 and 100>h >= 40 and ar >= 0.9 and ar <= 1.3:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    fourth_answers.append(c)
                
            elif ind==5:
                thresholded_image[5]=thresh
                if 100>w >= 40 and 100>h >= 40 and ar >= 0.8 and ar <= 1.3:
                    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
                    fifth_answers.append(c)
        # cv2.imshow("warped",cv2.resize(warped,dsize=None,fx=0.4,fy=0.3))
        # cv2.waitKey(0)
    return test_id,first_answers,second_answers,third_answers,fourth_answers,fifth_answers,thresholded_image


def check_answers(questions,paper,thresh,ANSWER_KEY):
    """Getting all ids of all answers form sheet"""
    answers={}
    questionCnts = contours.sort_contours(questions,
                                      method="top-to-bottom")[0]
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        get_answer=[0,0,0,0]
        bubbled = None
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            get_answer[j]=total
        answer=np.sort(get_answer)
        if answer[-1]>=850 and answer[-2]<850:
            answers[q+1]=np.argmax(get_answer)+1
        else:
            answers[q+1]=0
            
    return answers  
    
        

def get_id(questions,paper,thresh,TEST_ID):
    """Getting user's id from sheet"""
    id=[0,0,0,0,0,0,0,0]
    questionCnts = contours.sort_contours(questions,
                                      method="top-to-bottom")[0]
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(questions), 8)):
        incorrect_answers=0
        cnts = contours.sort_contours(questionCnts[i:i + 8])[0]
        bubbled = None
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total>900:
                if q==9:
                    q=-1
                id[j]=str(q+1)
    try:
        user_id="".join(id)
    except:
        user_id="empty"
    return user_id
    
