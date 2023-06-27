#Model Implementation

from data_preprocessing import splitting_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

X_train, X_validation, Y_train, Y_validation = splitting_dataset()


def fit_logistic_regression(X_train, X_validation, Y_train, Y_validation):
    
    #fit the model and get the expected output. our expected mean_score will be in range
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, Y_train)

    mean_score = model.score(X_validation, Y_validation)

    y_pred = model.predict(X_validation)
    std_score = np.std(y_pred == Y_validation)

    return ('Logistic Regression',mean_score, std_score)
    
    #model_logreg, mean_score, std_score = fit_logistic_regression(X_train, X_validation, Y_train, Y_validation)


def fit_lda(X_train, X_validation, Y_train, Y_validation):
    #fit the model and get the expected output. our expected mean_score will be in range
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)

    mean_score = model.score(X_validation, Y_validation)

    y_pred = model.predict(X_validation)
    std_score = np.std(y_pred == Y_validation)
    
    return ('Linear Discriminant Analysis', mean_score, std_score)
    
    #model_lda, mean_score, std_score = fit_lda(X_train, X_validation, Y_train, Y_validation)


def fit_gaussian_nb(X_train, X_validation, Y_train, Y_validation):
    #fit the model and get the expected output. our expected mean_score will be in range
    model = GaussianNB()
    model.fit(X_train, Y_train)

    # Evaluate the model on the validation set
    mean_score = model.score(X_validation, Y_validation)

    # Calculate the predictions
    y_pred = model.predict(X_validation)

    # Calculate the standard deviation of scores
    std_score = np.std(y_pred == Y_validation)

    return ('Gaussian Naive Bayes', mean_score, std_score)
    
    #model_gnb, mean_score, std_score = fit_gaussian_nb(X_train, X_validation, Y_train, Y_validation)



