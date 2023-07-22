import cv2 
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochs2.h5')


image = cv2.imread('C:\\Users\\Ayoub\\Desktop\\Cancer Detection with Python\\pred\\pred11.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img = np.expand_dims(img, axis=0)
probabilities = model.predict(input_img)
predicted_class = np.argmax(probabilities, axis=1)
print(predicted_class)


# input_img = np.expand_dims(img, axis=0)
# result = model.predict_classes(input_img)
# print(result)
