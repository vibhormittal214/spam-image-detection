# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:10:22 2019

@author: vibhor
"""

from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from PIL import Image
import numpy as np
depth = 3
width = 192
height = 192
shape= (depth, height, width)
if K.image_data_format() == 'channels_last':
    shape = (height, width, depth)
model = load_model('createdmodel.hdf5')
type_of_image = ['spam image-Shows Nuditity', 'partial spam image', 'spam image-shows adult image', 'spam image-violence', 'Not a spam image']
def main():
    img = Image.open('violence1.jpg')
    img1 = prepareimg(img)
    predict(img1)
    
def prepareimg(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((width, height))
    img1 = image.img_to_array(img)
    img1 = np.expand_dims(img1, axis=0)
    img1 /= 255. 
    return img1
def predict(test_image):
    pred_prob = model.predict(test_image)
    for index, prob in enumerate(pred_prob[0]):
        print(type_of_image[index],float(prob)*100,"%")
main()
