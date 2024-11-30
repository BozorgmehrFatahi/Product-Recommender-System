# Product-Recommender-System

# Recommendation System using Neural Networks

This repository contains a Python implementation of a recommendation system using deep learning techniques. The model predicts whether a user will like a product based on their previous ratings.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)

## Introduction

The goal of this project is to build a recommendation system that predicts user preferences for products based on historical rating data. We utilize embeddings for users and products to capture the latent features and relationships between them.

## Requirements

Ensure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- keras
- seaborn
- matplotlib

You can install them using pip.

## Data Preprocessing

1. Load the dataset from a CSV file.
2. Encode user and product IDs using LabelEncoder.
3. Scale the ratings using MinMaxScaler.
4. Create a binary classification target variable based on ratings (1 if rating >= 0.5, else 0).
5. Visualize the class distribution and apply Random Over Sampling to handle class imbalance.

## Model Architecture

The recommendation system utilizes a neural network model with the following components:

- Input layers for user and item IDs.
- Embedding layers for users and items to capture latent features.
- A dot product layer to compute interaction scores between user and item embeddings.
- Bias terms for users and items to improve the model's accuracy.
- A dense output layer with a sigmoid activation function for binary classification.

## Training the Model

The model is compiled using the Adam optimizer and binary crossentropy loss function. It is then trained on the training dataset with a specified number of epochs and validation split to monitor performance.

## Evaluation

After training, the model's performance is evaluated using metrics such as precision score and classification report. This helps in understanding the effectiveness of the recommendation system.
