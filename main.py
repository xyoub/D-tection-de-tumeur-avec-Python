import cv2
import os 
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical



image_directory='dataset/'

no_t_images=os.listdir(image_directory+ 'no/')
yes_t_images=os.listdir(image_directory+ 'yes/')

#print(no_t_images)
dataset=[]
label=[]

INPUT_SIZE=64
# function pour s avoire si la photo est une jpg ou pas en plus de la taille et les couleur 
for i , image_name in enumerate(no_t_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_t_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# dataset=np.array(dataset)
# label=np.array(label)

X_train, X_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

# print(X_train.shape)
# print(X_test.shape)

# print(y_train.shape)
# print(y_test.shape)

X_train=normalize(X_train, axis=1)
X_test=normalize(X_test, axis=1) 

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)

# Debut du model

model=Sequential()

model.add(Conv2D(32, (3,3) , input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))  
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# y_train = np.array(y_train)
# y_test = np.array(y_test)



model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(X_test, y_test), shuffle=False)


model.save('BrainTumor10Epochs2.h5')
