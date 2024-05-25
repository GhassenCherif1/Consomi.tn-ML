import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

target_size = (224, 224)

img = tf.keras.preprocessing.image.load_img("necklace5.jpg", target_size=target_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

img_test = np.array([img_array])
print(img_test.shape)
jewelleryModel = tf.keras.models.load_model("JewelleryModelV4.keras")
materialModel = tf.keras.models.load_model("MaterialModel.keras")
jewprediction_array = jewelleryModel.predict(img_test)
jewprediction = np.argmax(jewprediction_array,axis=1)
jewprediciton = jewprediction[0]
print(jewprediction_array)

if jewprediciton in [0,1,2]:
    matprediction_array = materialModel.predict(img_test)
    matprediction = np.argmax(matprediction_array,axis=1)
    matprediciton = matprediction[0]
    if(matprediciton == 0):
        print("gold")
    elif(matprediciton == 1):
        print("silver")
    elif(matprediciton == 2):
        print("bronze")
    if(jewprediciton==0):
        print("earring")
    elif(jewprediciton==1):
        print("necklace")
    elif(jewprediciton==2):
        print("ring")
elif jewprediciton==3:
    print("watch")

