# Classification (Computer Vision - BoVW)
# Image Classification using Bag of Visual Words

This repository contains Python code for applying the Bag of Visual Words technique to image classification using OpenCV libraries. The goal is to benchmark alternative implementations and evaluate their performance.

### Detailed Description

In this section, we will focus on applying the Bag of Visual Words (BoVW) technique to a dataset of your choice for image classification. The choosen dataset is [Mammals Classification](https://www.kaggle.com/datasets/anirudhg15/mammals-classification).

Additional requirements for the implementation include:

1. Using an alternative technique (e.g., ORB, SURF, BRISK, BRIEF, HOG) in addition to SIFT to describe points of interest. Provide a brief description of your chosen technique and how it works in the report.
2. Creating the dictionary using two different techniques. The first will be k-means (already given), and the second technique will be your choice (e.g., DBSCAN, MeanShift). Provide a brief description of your chosen clustering algorithm and how you calculated the words in the dictionary.
3. Conducting the experiments twice. The first time with an 80% training data and 20% test data ratio, and the second time with a 60% training data and 40% test data ratio.

The results of the experiments will be recorded in an Excel file with the following columns:

    Feature Extraction Algorithm | Clustering Technique | Train Data Ratio | Classifier Used | Accuracy (tr) | Precision (tr) | Recall (tr) | F1 Score (tr) | Accuracy (te) | Precision (te) | Recall (te) | F1 Score (te)

A comparative evaluation of the results from the individual experiments is presented in the report. Finally, we propose the best technique for the chosen dataset based on the evaluations.

Please note that the dataset we choose had a maximum of 1000 images (combined for train and test).

For your convenience, code tested in Python 3.6.7 is provided. The following versions of OpenCV were used:

    #pip install opencv-python==3.4.2.16
    #pip install opencv-contrib-python==3.4.2.16

---

This repository was initially created to store my personal Python codes but is also available to others interested in similar projects. The codes contained in this repo were specifically developed for a Machine Learning and Natural Language Processing course in my MSc program, as part of an image classification and data analysis project.

Please note that the data used in this repository belongs to their respective owners, and the repository's purpose is to showcase analytical skills and code implementation.
