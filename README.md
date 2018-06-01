# Deep Learning Image Classifier, Convolutional neural network (CNN)
Create a simple CNN model to classify 4 different classes of images (below)
- ID cards/passports
- slides
- paper documents
- receipts

# Details
- Preprocessing.py is used to pre-process images with just numpy and PIL
- Preprocessing.py processes images into 64x64x3 images in numpy array
- Preprocessing_resnet processes images into 200x200x3 images in numpy array
- matplotlib is used to view the images
- 1404 training data, around 300+ per class
- 64 testing data, around 10+ per class

# Testing
- 3 different models for transfer learning was used **VGG16**, **ResNet50** & **InceptionV3**.
- Vgg16 works the best among the 3.
- ResNet did no better than random guessing initially. Learning phase was set as 1 - training phase to improve performance.
- Performance only improved to 70+% accuracy which is not good.
- For inception v3, around 20 other top layers were unfrozen to be allowed to train resulting in better performance.
