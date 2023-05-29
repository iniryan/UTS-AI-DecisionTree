import os
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Set the path to the directory containing the images
image_dir = 'animals'

# Set the target labels/classes
classes = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison']

# Read the images and extract features
X = []
y = []

for class_label in classes:
    class_dir = os.path.join(image_dir, class_label)
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize the image to the input size of the CNN model
        image = preprocess_input(image)  # Preprocess the image according to the requirements of the CNN model
        X.append(image)
        y.append(class_label)

X = np.array(X)
y = np.array(y)

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using the VGG16 model
X_features = vgg_model.predict(X)

# Reshape the feature vectors
X_features = X_features.reshape(X_features.shape[0], -1)

# Save the features and labels to a CSV file
np.savetxt('features.csv', X_features, delimiter=',')
np.savetxt('labels.csv', y, delimiter=',', fmt='%s')
