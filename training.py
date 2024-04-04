import os
import cv2
import numpy as np

data_path = './fotos'

data_list = os.listdir(data_path)

labels =[]
face_data =[]
label = 0

for data in data_list:
    file_path = data_path + '/' + data

    for file in os.listdir(file_path):
        labels.append(label)

        face_data.append(cv2.imread(file_path+'/'+file,0))
        img = cv2.imread(file_path+'/'+file, 0)

    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer.create()
print("training")
print("Número de elementos en face_data:", len(face_data))
print("Número de elementos en labels:", len(labels))

face_recognizer.train(face_data, np.array(labels))

face_recognizer.write('my_face_recognizer_model.xml')