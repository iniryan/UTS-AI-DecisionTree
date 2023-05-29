import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using the VGG16 model
X_train_features = vgg_model.predict(X_train)
X_test_features = vgg_model.predict(X_test)

# Reshape the feature vectors
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Apply LDA for feature reduction
lda = LinearDiscriminantAnalysis(n_components=100)
lda.fit(X_train_features, y_train)
X_train_features = lda.transform(X_train_features)
X_test_features = lda.transform(X_test_features)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_features, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_features)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
