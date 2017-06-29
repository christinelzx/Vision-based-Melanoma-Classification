# Vision-based-Melanoma-Classification
A Deep Learning based classification of Melanoma trained on the ISIC 2016 dataset. 

cleanData.py is a preprocessing program that uses OpenCV to perform lesion segmentation using a combination of simple thresholding filters and canny edge detectors.

vgg16.py is based off of a previous implementation (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) which fine-tunes a pretrained vgg-16 network on a user-specific dataset through the Keras deep learning API. The input images to the learning model are the pre-processed images from the output of cleanData.py

To run cleanData.py format data according to the following:

data/dataset_1/cnn_data/balanced/test/processed/label_0
data/dataset_1/cnn_data/balanced/test/processed/label_1
data/dataset_1/cnn_data/balanced/train/processed/label_0
data/dataset_1/cnn_data/balanced/train/processed/label_1

or for the entire 'unbalanced' dataset (i.e. the number of benign examples is not equal to the number of malignant examples) use:
data/dataset_1/cnn_data/unbalanced/test/processed/label_0
data/dataset_1/cnn_data/unbalanced/test/processed/label_1
data/dataset_1/cnn_data/unbalanced/train/processed/label_0
data/dataset_1/cnn_data/unbalanced/train/processed/label_1


If your work benefits from this code please cite my paper! 
http://web.stanford.edu/~kalouche/docs/Vision_Based_Classification_of_Skin_Cancer_using_Deep_Learning_(Kalouche).pdf
Kalouche, Simon. "Vision-Based Classification of Skin Cancer using Deep Learning." 2016.



