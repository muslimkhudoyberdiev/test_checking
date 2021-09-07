from imutils.perspective import four_point_transform
import numpy as np
from imutils import contours
import pytesseract
import imutils,time
import cv2
# from check_result import ANSWER_KEY_1,ANSWER_KEY_2,ANSWER_KEY_3,ANSWER_KEY_4,ANSWER_KEY_5,TEST_ID


def read_image(img):
    """Read image from file"""
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
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
        # if 0.14*max_width<=w<=0.18*max_width and 0.23*max_height<=h<=0.28*max_height or 0.13*max_width<=w<=0.18*max_width and 0.43*max_height<=h<=0.48*max_height or 0.14*max_height<=h<=0.18*max_height and  0.14*max_width<=w<=0.18*max_width:
        if 0.14*max_width<=w<=0.18*max_width and 0.18*max_height<=h<=0.28*max_height or 0.14*max_width<=w<=0.18*max_width and 0.32*max_height<=h<=0.5*max_height:
            # paper=image[y:y+h,x:x+w]
            warped=gray_image[y:y+h,x:x+w]
            image_to_ocr=warped[Y:Y+int(0.16*h),X:X+int(0.14*max_width)]
            test_id=warped[int(0.20*h):y+h,int(0.001*w):x+w]
            if 0.32*max_height<=h<=0.5*max_height:
                warped=warped[int(0.15*h):y+h,int(0.2*w):x+w]
            else:
                warped=warped[int(0.25*h):y+h,int(0.2*w):x+w]
            results=pytesseract.image_to_string(image_to_ocr).split()
            if "1" in results:
                contours[1]=warped
                # origin_image[0]=paper
            elif "2" in results:
                contours[2]=warped
                # origin_image[1]=paper
            elif "3" in results:
                contours[3]=warped
                # origin_image[2]=paper
            elif "4" in results:
                contours[4]=warped
                # origin_image[3]=paper
            elif "5" in results:
                contours[5]=warped
                # origin_image[4]=paper
            else:
                contours[0]=test_id
                # origin_image[5]=test_id
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
                if 100>w >= 20 and 100>h >= 25 and ar >= 0.7 and ar <= 1.3:
                    test_id.append(c)
                
            if ind==1:
                thresholded_image[1]=thresh
                if 50>w >= 20 and 50>h >= 25 and ar >= 0.8 and ar <= 1.3:
                    first_answers.append(c)
                    
                
            elif ind==2:
                thresholded_image[2]=thresh
                if 50>w >= 20 and 50>h >= 25 and ar >= 0.8 and ar <= 1.3:
                    second_answers.append(c)
               
            elif ind==3:
                thresholded_image[3]=thresh
                if 50>w >= 20 and 50>h >= 25 and ar >= 0.8 and ar <= 1.3:
                    third_answers.append(c)
               
            elif ind==4:
                thresholded_image[4]=thresh
                if 50>w >= 20 and 50>h >= 25 and ar >= 0.8 and ar <= 1.3:
                    fourth_answers.append(c)
                
            elif ind==5:
                thresholded_image[5]=thresh
                if 50>w >= 20 and 50>h >= 25 and ar >= 0.8 and ar <= 1.3:
                    fifth_answers.append(c)
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
        answer=max(get_answer)
        if answer>400:
            answers[q+1]=np.argmax(get_answer)+1
        else:
            answers[q+1]=0
            
    return answers  
    
        

def get_id(questions,paper,thresh,TEST_ID):
    """Getting user's id from sheet"""
    id=[0,0,0,0,0,0,0]
    questionCnts = contours.sort_contours(questions,
                                      method="top-to-bottom")[0]
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(questions), 7)):
        incorrect_answers=0
        cnts = contours.sort_contours(questionCnts[i:i + 7])[0]
        bubbled = None
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total>450 and incorrect_answers<=1:
                if q==9:
                    q=-1
                id[j]=str(q+1)
                incorrect_answers+=1
            elif incorrect_answers>=2:
                id="Wrong ID"
    try:
        user_id="".join(id)
    except:
        user_id="empty"
    return user_id
    
