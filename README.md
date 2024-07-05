# Cancer Classifier

## Overview
In this project, we use Python libraries to create a K-Nearest Neighbor classifier that predicts whether a patient has breast cancer.

## Tasks

### Explore the Data
1. Import the breast cancer data using `load_breast_cancer` from `sklearn.datasets`.
2. Print the first data point (`breast_cancer_data.data[0]`) and feature names (`breast_cancer_data.feature_names`).
3. Print the target (`breast_cancer_data.target`) and target names (`breast_cancer_data.target_names`).

### Splitting the Data into Training and Validation Sets
4. Import `train_test_split` from `sklearn.model_selection`.
5. Split the data (`breast_cancer_data.data`) and labels (`breast_cancer_data.target`) into training and validation sets with `test_size=0.2` and `random_state=100`.
6. Store the training and validation sets and labels in variables (`training_data`, `validation_data`, `training_labels`, `validation_labels`).
7. Print the lengths of `training_data` and `training_labels` to confirm the split.

### Running the Classifier
8. Import `KNeighborsClassifier` from `sklearn.neighbors`.
9. Create a `KNeighborsClassifier` with `n_neighbors=3`.
10. Train the classifier using `fit` with `training_data` and `training_labels`.
11. Print the accuracy of the classifier on the validation set using `score` with `validation_data` and `validation_labels`.
12. Use a for loop to iterate over values of `k` from 1 to 100, train the classifier, and print validation accuracies. Determine the best `k`.

### Graphing the Results
13. Import `matplotlib.pyplot` as `plt`.
14. Create a list `k_list` containing values from 1 to 100 using `range`.
15. Create an empty list `accuracies`. Append validation accuracies to `accuracies` inside the for loop.
16. Plot `k_list` vs `accuracies` using `plt.plot`.
17. Set x-axis label to "k" (`plt.xlabel`), y-axis label to "Validation Accuracy" (`plt.ylabel`), and title to "Breast Cancer Classifier Accuracy" (`plt.title`).
18. Display the plot using `plt.show`.

## Further Exploration
- Experiment with different `random_state` values to observe variance in the results.

![image](https://github.com/XENO2410/Cancer-Classifier/assets/97669140/c7c0102b-e7bc-469b-acd7-5ce132f57fd4)

