import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Print the first data point and feature names
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

# Print target values and target names
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# Split data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# Print lengths of training data and labels
print(len(training_data))
print(len(training_labels))

# Uncomment the following block if you want to train and evaluate the classifier with k=3
'''
# Initialize KNN classifier with k=3
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training_data, training_labels)

# Calculate accuracy on validation set
score = classifier.score(validation_data, validation_labels)
print(score)
'''

# Initialize empty list to store accuracies
accuracies = []

# Iterate over values of k from 1 to 100
for k in range(1, 101):
    # Initialize KNN classifier with current value of k
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)

    # Calculate accuracy on validation set
    score = classifier.score(validation_data, validation_labels)
    accuracies.append(score)
    # print(score)  # Uncomment to print individual scores
  
# Create a list of k values from 1 to 100
k_list = list(range(1, 101))

# Plot the validation accuracies against k values
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
