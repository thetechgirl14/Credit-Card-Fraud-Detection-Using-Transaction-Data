#Import necessary modules and packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset():
    #loading dataset
    dataset = pd.read_csv('creditcard_data.csv')
    return dataset

def dataframe_shape():
    #checking its shape
    dataset = load_dataset()  
    shape = dataset.shape
    return shape

def sum_of_null_values():
	# Adding null values
    sumofnull = dataset.isnull().sum()
    return sumofnull

def check_datatypes():
    # Getting data types
    dataset = load_dataset()
    return dataset.dtypes

def data_decribe():
    #get summary statistics for the dataset
    data = dataset.describe()
    return data

def check_count_of_target_variable():
    #get the count of classes in the target variable
    data = dataset['Class'].value_counts()
    return data

def corr_matrix():
    #write a code to plot correlation matrix for the dataset and return the plot and correlation data
    #Do not delete these three predefined variable
    numeric_features = dataset.select_dtypes(include=[float, int])
    corr_data = numeric_features.corr()
    g = sns.heatmap(corr_data, cmap='coolwarm', annot=True)
    plt.title('Correlation Matrix')
    plt.show()
    return corr_data

def plot_target_count():
    #plot using countplot on target 'Class' variable
    #Don't delete this predefined variable. write a code for plotting using that variable
    ax = sns.countplot(x='Class', data=dataset)
    plt.title('Countplot of Target Variable')
    plt.show()
    return ax
