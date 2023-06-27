#Data Preprocessing

import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_exploration import load_dataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


#Feature Scaling needs to be done for "Amount" column in our dataset. use standardscaler

def feature_scaling_amount():
    #return the updated dataset where the 'Amount' column values are scaled
    data = load_dataset()
    sc = StandardScaler()
    data['Amount'] = sc.fit_transform(data['Amount'].values.reshape(-1, 1))
    return data


#Drop unnecessary columns. In our dataset time column has no significance. so we can drop that.

def drop_unnecessary_columns():
    #return the updated dataset where the 'Time' column is dropped
    dataset = feature_scaling_amount()
    dataset.drop('Time', axis=1, inplace=True)
    return dataset


#Lets check for the duplicate rows count

def drop_duplicate_data():
    #return the updated dataset where the dupicate rows are dropped
    dataset = drop_unnecessary_columns()
    data = dataset.drop_duplicates()
    return data


#Feautures separating
# In x all the variable should be present ,except target variable
#In this section, the 'target' (dependent) column will be seperated from independent columns.



def feature_separating_x_y():
	dataset = drop_duplicate_data()
	X = dataset.drop('Class', axis = 1)
	y = dataset['Class']
	return X,y


#Handling Imbalanced Dataset
#Target variable is highly imbalanced. To balance it before fitting in the machine learning models.
#In this project, Oversampling technique(SMOTE) has been used to balance the dataset


def data_balancing_smote():
    #return the updated dataset where the data is balanced
    x,y = feature_separating_x_y()
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(x, y)
    return X_res, y_res


#Splitting The Dataset Into The Training Set And Test Set
#The dataset will be splitted into 80:20 ratio (80% training and 20% testing).

def splitting_dataset():
	X_res, y_res = data_balancing_smote()
	x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
	return x_train, x_test, y_train, y_test