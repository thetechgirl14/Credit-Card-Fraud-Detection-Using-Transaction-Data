# Credit Card Fraud Detection using Transaction Data

This project aims to detect credit card fraud using transaction data. It utilizes various machine learning techniques to preprocess the data, implement different models, evaluate their performance, and make predictions on new data.

## Project Structure

The project consists of the following files:

[data_exploration.py](https://github.com/thetechgirl14/Credit-Card-Fraud-Detection-Using-Transaction-Data/blob/main/data_exploration.py): This file contains functions to explore and analyze the dataset. It includes functions to load the dataset, check its shape, sum of null values, data types, summary statistics, correlation matrix, and plot the count of the target variable.

[data_preprocessing.py](https://github.com/thetechgirl14/Credit-Card-Fraud-Detection-Using-Transaction-Data/blob/main/data_preprocessing.py): This file focuses on preprocessing the data before model implementation. It includes functions for feature scaling, dropping unnecessary columns, removing duplicate rows, and separating the features and target variables. It also handles imbalanced datasets using the SMOTE technique and splits the dataset into training and testing sets.

[model_implementation.py](https://github.com/thetechgirl14/Credit-Card-Fraud-Detection-Using-Transaction-Data/blob/main/model_implementation.py): This file implements different machine learning models for credit card fraud detection. It includes functions to fit logistic regression, linear discriminant analysis, and Gaussian Naive Bayes models on the training data.

[model_evaluation.py](https://github.com/thetechgirl14/Credit-Card-Fraud-Detection-Using-Transaction-Data/blob/main/model_evaluation.py): This file evaluates the performance of the implemented models. It compares the mean scores of the models and selects the best one. It then fits the best model on the training data, makes predictions on the validation data, and calculates evaluation metrics such as accuracy, confusion matrix, and classification report.

[prediction.py](https://github.com/thetechgirl14/Credit-Card-Fraud-Detection-Using-Transaction-Data/blob/main/prediction.py): This file demonstrates how to use the implemented models for making predictions on new data. It calls the evaluate_model function from model_evaluation.py to obtain the final results. It also includes a sample data input and outputs the predicted class label.

## Usage
To use this project:

1. Make sure you have the necessary dependencies installed
```
pip install -r requirements.txt
```

2. Download the dataset named "creditcard_data.csv" and place it in the same directory as the Python files.

3. Import the required functions from the respective files to perform specific tasks. For example, to load the dataset and check its shape, use the following code:

```
from data_exploration import load_dataset, dataframe_shape

dataset = load_dataset()
shape = dataframe_shape()

print("Dataset shape:", shape)
```
4. Follow similar steps to perform data preprocessing, implement different models, evaluate their performance, and make predictions on new data.

Note: Modify the code as per your specific requirements and customize it accordingly.

Conclusion
This project provides a comprehensive solution for credit card fraud detection using transaction data. By following the steps outlined in the files, you can preprocess the data, implement different models, evaluate their performance, and make predictions on new data. Feel free to explore and modify the code to fit your specific needs and enhance the fraud detection capabilities.
