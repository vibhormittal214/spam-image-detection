# spam-image-detection

NFSW(Not suitable for work-spam images) andd SFW(Suitable for work) image detection
This repository contains python code for identifying spam images.

You can see the video attached to see how the image is being predicted

Note- First unzip createdmodel and add the main file(createdmodel.hdf5) to the main directory-i.e where the directory where the code is saved.

Conatins createdmodel.hdf5 which has the saved trained model and you can directly use it to test the image. Simply open spam_image_predict.py and pass the 'image_name.format' in Image.open()

Spam Images- Images that may contain adult content, Nuditity, or violence images(These images are called NOT SUITABLE FOR WORK). Download the spam images(NFSW IMAGE) and save to directory containing the code and test the model

Not spam images- Those are suitable for work

The cnn model is implemented to detect spam images.

The data set for training the model- https://www.dropbox.com/s/opiqoh550jd1glb/dataset.zip?dl=0 (This dataset is not mine and contains spam images and not spam images and this dataset is used for educational purpose only)

1) Cnn_train- for training the model. As the data set is very large the model will take a long time to train. So The trained model is attached with the repository and you can directly use it to test the images.

2) spam_image_predict- It uses the saved trained model for predicting whether the image is a spam image or not. You have to save the image to be tested in the same directory and then pass this-> 'image_name.format' in Image.open()
