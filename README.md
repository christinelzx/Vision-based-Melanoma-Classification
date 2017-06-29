# Vision-based-Melanoma-Classification
A Deep Learning based classification of Melanoma trained on the ISIC 2016 dataset. 

cleanData.py is a preprocessing program that uses OpenCV to perform lesion segmentation using a combination of simple thresholding filters and canny edge detectors.

vgg16.py is based off of a previous implementation (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) which fine-tunes a pretrained vgg-16 network on a user-specific dataset through the Keras deep learning API. The input images to the learning model are the pre-processed images from the output of cleanData.py

If your work benefits from this code please cite my paper! 

http://web.stanford.edu/~kalouche/docs/Vision_Based_Classification_of_Skin_Cancer_using_Deep_Learning_(Kalouche).pdf

Kalouche, Simon. "Vision-Based Classification of Skin Cancer using Deep Learning." 2016.



