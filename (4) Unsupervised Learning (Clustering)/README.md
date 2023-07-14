# Unsupervised Learning (Clustering)
## Fashion-MNIST Image Clustering and Deep Learning

This repository contains Python code for developing combined deep learning models and clustering techniques on the Fashion-MNIST dataset. The goal is to compare clustering results when using raw data versus complex descriptive features extracted using stacked autoencoders.

## Detailed Description

In this repository, we will focus on developing code that showcases the differences in clustering results when using different approaches on the Fashion-MNIST dataset. The dataset can be found at the following link: [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist).

The code developed here includes the following steps:

1. Loads the Fashion-MNIST data.
2. Separates the data into three sets: training, validation, and test data.
3. Creates a CNN autoencoder that compresses and recreates the image input.
4. Randomly prints images from the dataset (one from each class) and their reconstructed versions as provided by the autoencoder.
5. Uses the encoder part of the autoencoder to encode the images.
6. Applies three different clustering techniques of our choice to create subgroups.
7. Calculates performance indicators presented in the lab, as well as an additional indicator of our choice.
8. Presents indicative clustering results for random images.
9. Performs steps 6, 7, and 8 using both the pixel values of the images (normalized to [0, 1]) and the values produced by the encoder part of the CNN autoencoder.
10. Utilizes the algorithm results and graphs created in Excel to produce a report that presents conclusions, performs comparative evaluations, and suggests the best possible technique for the particular case.

---

This repository was initially created to store personal Python codes but is also available to others interested in similar projects. The codes contained in this repository were specifically developed for a Machine Learning and Natural Language Processing course in the MSc program, as part of an image clustering and deep learning project.

Please note that the data used in this repository belongs to their respective owners, and the repository's purpose is to showcase analytical skills and code implementation.
