# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:48:36 2019

@author: DELL
"""
#-----------PART1--------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#initailize CNN
classifier=Sequential()#making object
#step1-convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))#as we are working on CPU so go for 32


#step2-pooling-to reduce the size using max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding 2nd convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))#as we are working on CPU so go for 32
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step3-Flattening -to convert to one single vector

classifier.add(Flatten())#keras will automatically get the parameter from the previous values.



#step 4- full connection
classifier.add(Dense(output_dim = 128,activation ='relu'))
classifier.add(Dense(output_dim = 1,activation ='sigmoid'))



#compliation
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])



#---------------------PART2--------------------
#image processing-that is fitting CNN into images.
#IMAGE AUGMENTATION----IT ALLOWS US TO GET GOOD RESULT WITH SMALL DATA
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

#PREDICTING THE IMAGE
import numpy as np
from keras.preprocessing import image
t_image=image.load_img('dataset/single_prediction/dog.4.jpg',target_size=(64,64))
t_image=image.img_to_array(t_image)
t_image=np.expand_dims(t_image,axis=0)
result=classifier.predict(t_image)

training_set.class_indices

if result[0][0] == 1:
    prediction= 'dog'
else:
    prediction= 'cat'






