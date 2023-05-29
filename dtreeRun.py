import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Load the features and labels from the CSV files
X_features = np.loadtxt('features.csv', delimiter=',')
y = np.loadtxt('labels.csv', delimiter=',', dtype=str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Apply PCA for feature reduction
pca = PCA(n_components=100)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_pca, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_pca)

# Convert labels to numeric values
unique_labels = np.unique(y)
label_to_numeric = {label: i for i, label in enumerate(unique_labels)}
y_test_numeric = np.array([label_to_numeric[label] for label in y_test])
y_pred_numeric = np.array([label_to_numeric[label] for label in y_pred])

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mse = mean_squared_error(y_test_numeric, y_pred_numeric)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Mean Squared Error:", mse)
