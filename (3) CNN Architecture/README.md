# CNN Architecture
## CIFAR-10 Image Classification with CNN

This repository contains Python code for developing a Convolutional Neural Network (CNN) model for image classification using the CIFAR-10 dataset. The goal is to train the model and evaluate its performance on the test data, comparing different experiments.

## Part A: TrainTheModel

In this section, we will focus on developing the `TrainTheModel` routine for training the CNN model. The following steps should be included:

1. Load the CIFAR-10 data and create the training and validation sets (note: do not confuse the validation and test sets).
2. Plot 4 random images per class based on the actual class.
3. Define the CNN architecture, including layers, loss functions, number of kernels, etc.
4. Perform the training and save the model.
5. Display relevant messages on the screen to inform about the status and progress of the code execution (e.g., data creation completed, training paradigms size, CNN topology setup completed, etc.).

## Part B: TestTheModel

In this section (separate script), we will focus on the `TestTheModel` routine for evaluating the trained CNN model. The following steps should be included:

1. Load the trained CNN model from the previous step.
2. Run the model on the test data and calculate metric scores such as accuracy, precision, recall, and F1-score for the test set.
3. Print a confusion matrix to visualize the performance of the model.
4. Plot 4 random images per category. Note that the category is now derived based on the predictions made by the CNN, not the actual class.

## Experiments

We will perform the above experiment (Experiment 1) two more times:

1. **Experiment 2**: Using the same architecture but a different loss function during training.
2. **Experiment 3**: Using a more complex structure, such as extra convolution layers or additional kernels.

The results of the experiments on the train and test sets will be compared and analyzed.

---

This repository was initially created to store personal Python codes but is also available to others interested in similar projects. The codes contained in this repository were specifically developed for a Machine Learning and Natural Language Processing course in the MSc program, as part of an image classification and data analysis project.

Please note that the data used in this repository belongs to their respective owners, and the repository's purpose is to showcase analytical skills and code implementation.
