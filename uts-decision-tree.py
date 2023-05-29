# Anggota Kelompok:
# 1. RYAN ADI SAPUTRA (1201210006)
# 2. SULTHONY AKBAR RIZKI PAMBUDI (1201210014)
# 3. YAFI YOGA ABID PRAMONO (1201210022)

# Import library
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Set the path to the directory containing the images
image_dir = 'animals'

# Set the target labels/classes
classes = ['antelope', 'bat', 'bee', 'bison', 'cat', 'crow', 'dog', 'elephant', 
           'lion', 'orangutan', 'panda', 'penguin', 'pigeon', 'snake', 'tiger',
           'whale', 'zebra']

# Read the images and extract features
X = []
y = []

for class_label in classes:
    class_dir = os.path.join(image_dir, class_label)
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        # Resize the image to the input size of the CNN model
        image = cv2.resize(image, (224, 224))  
        # Preprocess the image according to the requirements of the CNN model
        image = preprocess_input(image)  
        X.append(image)
        y.append(class_label)

X = np.array(X)
y = np.array(y)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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

# Apply PCA for feature reduction
pca = PCA(n_components=100)
pca.fit(X_train_features)
X_train_features = pca.transform(X_train_features)
X_test_features = pca.transform(X_test_features)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_features, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_features)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mse = mean_squared_error(y_test.astype(float), y_pred.astype(float))

# Print the evaluation metrics
print("Accuracy: {:.2%}".format(accuracy))
print("Precision: {:.2%}".format(precision))
print("Recall: {:.2%}".format(recall))
print("F1-Score: {:.2%}".format(f1))
print("Mean Squared Error:", mse)
