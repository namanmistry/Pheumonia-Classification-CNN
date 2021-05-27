import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import cv2

# data = []

# for file in os.listdir("./data/train/NORMAL"):
#     img = cv2.imread(f"./data/train/NORMAL/{file}")
#     img = cv2.resize(img,(200,200))
#     img_array = np.array(img).flatten()
#     data.append([img_array,0])

# count = 0

# for file in os.listdir("./data/train/PNEUMONIA"):
#     img = cv2.imread(f"./data/train/PNEUMONIA/{file}")
#     img = cv2.resize(img,(200,200))
#     img_array = np.array(img).flatten()
#     data.append([img_array,1])
#     if count > 1341:
#         break
#     count+=1

# pickle_open = open("data.pkl","wb")
# pickle.dump(data,pickle_open)
# pickle_open.close()

# pickle_open = open("data.pkl","rb")
# data = pickle.load(pickle_open)
# data = np.array(data)
# random.shuffle(data)
# features = []
# labels = []

# for feature, label in data:
#     features.append(feature)
#     labels.append(label)

# features = np.array(features)
# labels = np.array(labels)


# pickle_open = open("features.pkl","wb")
# pickle.dump(features,pickle_open)
# pickle_open.close()

# pickle_open = open("labels.pkl","wb")
# pickle.dump(labels,pickle_open)
# pickle_open.close()

# pickle_open = open("features.pkl","rb")
# features = pickle.load(pickle_open)

# features = features/255
# features = features.reshape(-1,200,200,3)

# pickle_open = open("labels.pkl","rb")
# labels = pickle.load(pickle_open)
# model=Sequential([
#     layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same',input_shape=(200,200,3)),
#     layers.MaxPool2D(pool_size=(2,2),strides=2),
#     layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'),
#     layers.MaxPool2D(pool_size=(2,2),strides=2),
#     layers.Flatten(),
#     layers.Dense(1,activation='sigmoid'),
# ])

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# model.fit(features, labels, epochs=10, batch_size=4)

# data_test = []

# for file in os.listdir("./data/test/NORMAL"):
#     img = cv2.imread(f"./data/test/NORMAL/{file}")
#     img = cv2.resize(img,(200,200))
#     img_array = np.array(img).flatten()
#     data_test.append([img_array,0])

# for file in os.listdir("./data/test/PNEUMONIA"):
#     img = cv2.imread(f"./data/test/PNEUMONIA/{file}")
#     img = cv2.resize(img,(200,200))
#     img_array = np.array(img).flatten()
#     data_test.append([img_array,1])

# pickle_open = open("test.pkl","wb")
# pickle.dump(data_test,pickle_open)
# pickle_open.close()

model = models.load_model("model.h5")

# pickle_open = open("test.pkl","rb")
# data = pickle.load(pickle_open)
# data_test = np.array(data)
# random.shuffle(data_test)

# feature_test = []
# label_test = []

# for feature, label in data_test:
#     feature_test.append(feature)
#     label_test.append(label)

# feature_test = np.array(feature_test)
# label_test = np.array(label_test)

# feature_test = feature_test/255
# feature_test = feature_test.reshape(-1,200,200,3)
# model.evaluate(feature_test,label_test)

# img = cv2.imread("test2.jpeg")
# img = cv2.resize(img, (200,200))
# img_array = np.array(img).flatten()

# img_array = img_array/255

# if (model.predict(img_array.reshape(1,200,200,3))) < 0.5:
#     print("NORMAL")
# else:
#     print("PNEUMONIA")
