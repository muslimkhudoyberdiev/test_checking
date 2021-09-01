from prepocessing.preprocess import read_image,separate_answers,separate_contours,get_id,check_answers
from prepocessing.check_result import *
import sys



def main(image):
    image,gray=read_image(image)
    frames=separate_contours(image,gray)
    test_id,first,second,third,fourth,fifth,thresh=separate_answers(frames)
    for i in range(0,6):
        if i==0:
            user_id=get_id(test_id,frames[i],thresh[i],TEST_ID)
            print(user_id)
        if i==1:
            print("FIRST_ANSWERS")
            first_answers=check_answers(first,frames[i],thresh[i],ANSWER_KEY_1)
            print(first_answers)
        if i==2:
            print("SECOND_ANSWERS")
            second_answers=check_answers(second,frames[i],thresh[i],ANSWER_KEY_2) 
            print(second_answers)
        if i==3:
            print("THIRD_ANSWERS")
            third_answars=check_answers(third,frames[i],thresh[i],ANSWER_KEY_3) 
            print(third_answars)
        if i==4:
            print("FOURTH_ANSWERS")
            fourth_answers=check_answers(fourth,frames[i],thresh[i],ANSWER_KEY_4)
            print(fourth_answers)
        if i==5:
            print("FIFTH_ANSWERS")
            fifth_aswers=check_answers(fifth,frames[i],thresh[i],ANSWER_KEY_5) 
            print(fifth_aswers)

main(sys.argv[1])