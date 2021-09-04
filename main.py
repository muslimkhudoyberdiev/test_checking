from prepocessing.preprocess import read_image,separate_answers,separate_contours,get_id,check_answers
from prepocessing.check_result import *
import sys



def main(image):
    data={}
    image,gray=read_image(image)
    frames=separate_contours(image,gray)
    test_id,first,second,third,fourth,fifth,thresh=separate_answers(frames)
    for i in range(0,6):
        if i==0:
            user_id=get_id(test_id,frames[i],thresh[i],TEST_ID)
            data["user_id"]=user_id
        if i==1:
            first_answers=check_answers(first,frames[i],thresh[i],ANSWER_KEY_1)
            data["first"]=first_answers
        if i==2:
            second_answers=check_answers(second,frames[i],thresh[i],ANSWER_KEY_2) 
            data["second"]=second_answers
        if i==3:
            third_answers=check_answers(third,frames[i],thresh[i],ANSWER_KEY_3) 
            data["third"]=third_answers
        if i==4:
            fourth_answers=check_answers(fourth,frames[i],thresh[i],ANSWER_KEY_4)
            data["fourth"]=fourth_answers
        if i==5:
            fifth_aswers=check_answers(fifth,frames[i],thresh[i],ANSWER_KEY_5) 
            data["fifth"]=fifth_aswers
    return [data]
main(sys.argv[1])