import numpy as np
import mediapipe as mp
import cv2 as cv
import tensorflow as tf
print(cv.__version__)
model=tf.keras.models.load_model("wave_bot.h5")
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
while True:
    ret, img=cam.read()
    if not ret:
        break
    key=cv.waitKey(1)
    if(key == ord("q")):
        break
    data,img=give_cords(img)
    if len(data) == 63:
        
        prediction=model.predict(np.array(data).reshape(1,-1))[0][0]
        if prediction >0.5:
            msg="stop"
        else:
            msg="move"
        cv.putText(img,msg,(300,50),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
    else:
        print("pls keep exactly one hand in frame")
    cv.imshow("image",img)
cam.release()
cv.destroyAllWindows()