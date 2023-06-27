#final prediction
#The goal is to accurately predict whether the transaction is normal[0] or fraud[1].

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from model_implementation import fit_logistic_regression, fit_lda, fit_gaussian_nb
from data_preprocessing import splitting_dataset
from model_evaluation import evaluate_model


def predict_model(data):

    X_train, X_validation, Y_train, Y_validation = splitting_dataset()
    
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
    
    # Make a prediction on the new data using the your model
    prediction = model.predict(data)

    # Return the predicted class label
    return prediction


if __name__ == '__main__':

    # Call the evaluate_model function to get the final results
    accuracy, cm, cr = evaluate_model()

    # Print the results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    data = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]  ## thal_2, thal_3, slope_0, slope_1, slope_2

    predictions = predict_model(data)
    print(predictions)