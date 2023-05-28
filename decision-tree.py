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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Set the path to the directory containing the images
image_dir = 'animals'

# Set the target labels/classes
classes = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
           'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
           'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
           'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus',
           'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard',
           'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter',
           'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine',
           'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark',
           'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle',
           'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

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

# Preprocess the labels
y_encoded = np.zeros(y.shape, dtype=int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using the VGG16 model
X_train_features = vgg_model.predict(X_train)
X_test_features = vgg_model.predict(X_test)

# Reshape the feature vectors
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Apply PCA or LDA for feature reduction
# Uncomment one of the following sections based on your choice

# # Apply PCA for feature reduction
# pca = PCA(n_components=100)
# pca.fit(X_train_features)
# X_train_features = pca.transform(X_train_features)
# X_test_features = pca.transform(X_test_features)

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
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mse = mean_squared_error(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Mean Squared Error:", mse)
