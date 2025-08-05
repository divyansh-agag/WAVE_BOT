import numpy as np
import mediapipe as mp
import cv2 as cv
print(cv.__version__)
cam=cv.VideoCapture(0)
height=480

weidth=640
cam.set(cv.CAP_PROP_FPS,10)

cam.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc(*'MJPG'))
hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=.5,min_tracking_confidence=.5)
mpdraw=mp.solutions.drawing_utils
def give_cords(img):
    img=cv.resize(img,(weidth,height))
    results=hands.process(img)
    train=[]
    if results.multi_hand_landmarks!=None:    
        for handlandmarks in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlandmarks,mp.solutions.hands.HAND_CONNECTIONS)

            train_x=[]##
            #print(f"len:{len(handlandmarks.landmark)}")##21
            for landmarks in handlandmarks.landmark:
                train_x.append(int(landmarks.x*weidth))
                train_x.append(int(landmarks.y*height))
                train_x.append(landmarks.z)
            train=train_x
    return train,img
    
training_x=[]
training_y=[]
##1 open
##0 close
pose2_st=False
training=False
while True:
    ret, img=cam.read()
    if not ret:
        break
    train,img=give_cords(img)
    key=cv.waitKey(100)
    
    if(len(train)!=0) and training == True and pose2_st == False:
        training_x.append(train)
        training_y.append(1)
    if(len(train)!=0) and training == True and pose2_st == True:
        training_x.append(train)
        training_y.append(0)
    if key==ord('q'):
        
        print(len(training_y))
        np.save("train_x",np.array(training_x))
        np.save("train_y",np.array(training_y))
        break
    if key==ord('p'):
        print(f"current size of dataset: {len(training_y)}")
        training=False
    if key==ord('s'):
        training=True
    if key == ord("c"):
        training=False
        pose2_st=True
    
    cv.imshow("my window",img)
    cv.moveWindow("my window",0,0)
    
    
cam.release()
cv.destroyAllWindows()
