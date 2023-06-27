#Finalize model and Evaluation metrics
# In this file, to choose (finalize) the best model with the help of mape and then calculate the evaluation metric:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from model_implementation import fit_logistic_regression, fit_lda, fit_gaussian_nb
from data_preprocessing import splitting_dataset

X_train, X_validation, Y_train, Y_validation = splitting_dataset()

def evaluate_model():
    # Fit the logistic regression model
    model_logreg, mean_score_logreg, std_score_logreg = fit_logistic_regression(X_train, X_validation, Y_train, Y_validation)
    
    # Fit the linear discriminant analysis model
    model_lda, mean_score_lda, std_score_lda = fit_lda(X_train, X_validation, Y_train, Y_validation)
    
    # Fit the Gaussian Naive Bayes model
    model_gnb, mean_score_gnb, std_score_gnb = fit_gaussian_nb(X_train, X_validation, Y_train, Y_validation)
    
    # Compare the mean scores of the models and choose the best one
    if mean_score_logreg > mean_score_lda and mean_score_logreg > mean_score_gnb:
        best_model = model_logreg
    elif mean_score_lda > mean_score_logreg and mean_score_lda > mean_score_gnb:
        best_model = model_lda
    else:
        best_model = model_gnb
    
    # Fit the best model on the training data
    if best_model == model_logreg:
        model = LogisticRegression(random_state=42)
    elif best_model == model_lda:
        model = LinearDiscriminantAnalysis()
    else:
        model = GaussianNB()
    
    model.fit(X_train, Y_train)
    
    # Make predictions on the validation data
    y_pred = model.predict(X_validation)
    
    # Calculate the accuracy score
    accuracy = accuracy_score(Y_validation, y_pred)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(Y_validation, y_pred)
    
    # Calculate the classification report
    cr = classification_report(Y_validation, y_pred)
    
    return accuracy, cm, cr

