# Consomi.tn ML: Machine Learning for E-Commerce

## Description

This repository contains the machine learning components of the Consomi.tn e-commerce platform. The goal is to develop models that can identify products from images and predict their prices based on various features. It also includes a Flask API for communication with the backend.

## Features

- **Product Identification**: Use image recognition to classify products from uploaded photos.
- **Price Prediction**: Predict the selling price of products based on identified features and additional information provided.
- **Flask API**: A RESTful API built with Flask to interface with the backend and provide predictions.

## Components

1. **Data Collection**:
   - Scripts and notebooks for gathering and preprocessing image data using Techniques like web scrapping and data mining ...
   - Methods for collecting labeled datasets for training and evaluation.

2. **Image Recognition**:
   - Models and algorithms for product identification using deep learning techniques.
   - Training scripts for CNNs models such as MobileNet, EfficientNet, ResNet ...

3. **Feature Extraction**:
   - Techniques for extracting relevant features from images and forms.
   - Data preprocessing steps to enhance model performance.

4. **Price Prediction**:
   - Regression models to estimate the price of the identified products using many algorithms like decision trees, neural networks, xgboost ...
   - Evaluation metrics and methods for improving prediction accuracy.

5. **Flask API**:
   - A RESTful API to handle requests for product identification and price prediction.
   - Endpoints for uploading images and retrieving predictions.


## Run Locally

Clone the project

```bash
  git clone https://github.com/GhassenCherif1/Consomi.tn-ML.git
```

Go to the project directory

```bash
  cd Consomi.tn-ML
```

Install packages

```bash
  pip install -r requirements.txt
```

