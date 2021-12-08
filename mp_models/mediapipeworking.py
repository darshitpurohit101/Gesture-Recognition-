import cv2 as cv

import numpy as np
import pandas as pd

import tensorflow as tf

import datetime
now = datetime.datetime.now()

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import time


def tick():
    image=np.zeros((600,800,3))
    cv.rectangle(image, (0,0), (800,600), (0,0,0), cv.FILLED)
    cv.imshow("AYAYAYAYAYAYAYA", image)
    key=cv.waitKey()
    if key == 114: #key = 'r'
        train_mode()
    elif key == 115: #key = 's'
        test_mode()
    elif key == 118: #key = 'v'
        train_vid_mode()
    elif key == 27:
        final = True
        cv.destroyWindow("AYAYAYAYAYAYAYA")

def landmark_to_data(landmarks):
    coord_list=np.array([])
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z
        coord_list = np.append(coord_list,np.array([landmark_x,landmark_y,landmark_z]))
    return coord_list

def dat_formatting(dat):
    pass


def train_vid_mode():
    dat_l=[]
    cap = cv.VideoCapture("darshit.MOV")


    prevTime = 0

    dat = np.zeros((1,63))
    lab = np.array([])
    with mp_hands.Hands(
        min_detection_confidence=0.5,       #Detection Sensitivity
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv.putText(image, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv.imshow('MediaPipe Hands', image)

        gotkey = cv.waitKey(30)

        if gotkey == 27:
            l=[[f'x{i}',f'y{i}',f'z{i}'] for i in range(21)]
            ln=[item for sublist in l for item in sublist]
            dat_l=pd.DataFrame(dat[1:], columns = ln)
            print(lab)
            dat_l['label']=lab
            dat_l.to_csv('dat_'+now.strftime("%m_%d_%H%M")+'.csv')
            cv.destroyWindow('MediaPipe Hands')
            return dat_l

        elif gotkey !=-1:
            #write labeled hand_landmarks to a data file
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #print(hand_landmarks)
                    print(landmark_to_data(hand_landmarks))
                    print(type(gotkey))
                    newrow = landmark_to_data(hand_landmarks)
                    dat = np.vstack((dat, newrow))

                    lab = np.append(lab,gotkey)
                    print(lab)

    cap.release()


def train_mode():
    dat_l=[]
    cap = cv.VideoCapture(0)

    prevTime = 0

    dat = np.zeros((1,63))
    lab = np.array([])
    with mp_hands.Hands(
        min_detection_confidence=0.5,       #Detection Sensitivity
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv.putText(image, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv.imshow('MediaPipe Hands', image)

        gotkey = cv.waitKey(30)

        if gotkey == 27:
            l=[[f'x{i}',f'y{i}',f'z{i}'] for i in range(21)]
            ln=[item for sublist in l for item in sublist]
            dat_l=pd.DataFrame(dat[1:], columns = ln)
            print(lab)
            dat_l['label']=lab
            dat_l.to_csv('dat_'+now.strftime("%m_%d_%H%M")+'.csv')
            cv.destroyWindow('MediaPipe Hands')
            return dat_l

        elif gotkey !=-1:
            #write labeled hand_landmarks to a data file

            for hand_landmarks in results.multi_hand_landmarks:
                #print(hand_landmarks)
                print(landmark_to_data(hand_landmarks))
                print(type(gotkey))
                newrow = landmark_to_data(hand_landmarks)
                dat = np.vstack((dat, newrow))

                lab = np.append(lab,gotkey)
                print(lab)

    cap.release()


def test_mode():
    dat_l=[]
    cap = cv.VideoCapture(0)

    prevTime = 0

    dat = np.zeros((1,63))
    lab = np.array([])
    letlist=['0','0','0','0','0']
    with mp_hands.Hands(
        min_detection_confidence=0.5,       #Detection Sensitivity
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv.putText(image, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)


        gotkey = cv.waitKey(30)

        if gotkey == 27:
            cv.destroyWindow('MediaPipe Hands')
            return 0

        else:
            #Classify hand gesture
            if results.multi_hand_landmarks!=None:
                for hand_landmarks in results.multi_hand_landmarks:

                    sample = landmark_to_data(hand_landmarks)
                    tf_sample=tf.convert_to_tensor(sample)
                    tf_sample_good=tf.reshape(tf_sample,[1,63])

                    pred=model.predict(tf_sample_good)
                    if np.argmax(pred)!=26:
                        letter=chr(np.argmax(pred)+97)
                    else:
                        letter='sp'
                    letlist.pop(0)
                    letlist.append(letter)
                    print(letter)

            let_avg = max(set(letlist), key = letlist.count)
            cv.putText(image, 'Pred: '+let_avg, (420, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            cv.imshow('MediaPipe Hands', image)

    cap.release()



#______..._.___:2'':EÉßß_main______..._.___:2'':EÉßß_:

model = tf.keras.models.load_model('saved_models/newmodel')
tf.keras.utils.plot_model(model, to_file='modelplot.png', show_shapes=False)

print("Program Starting")
while True:
        final = False
        tick()
        if final == True:
            quit()
        time.sleep(2)
