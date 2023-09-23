# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
# Importing Libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset
df=pd.read_csv('Churn_Modelling.csv')
df

#Checking for null values
df.isnull().sum()

#Checking for dulpicated values
df.duplicated()

#Dropping unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df

#Normalising using MinMaxScaler
ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2

#Splitting the dataset - x
X=df2.iloc[:,:-1].values
X

#Splitting the dataset - y
y=df2.iloc[:,-1].values
y

# Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))

```
## OUTPUT:

Read Dataset

![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/da08c00f-326d-4ebe-bf3f-d6b26841f404)


Checking for Null values

![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/1644ff14-c5de-4c6a-9c5d-d2c956780e41)


Checking for duplicated values

![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/37d52aa2-0d11-4bff-98ff-e5d7acfdfe1d)


Dropping off unwanted values


![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/96500729-3d55-4c3d-9b01-88584320e238)


Normalised data using MinMaxScaler


![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/844a96fa-42c8-484b-8edb-5315f8236f3a)


Split values of X dataset


![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/442e82e7-c649-4eb6-afd3-5ef6133a6715)


Split values of y dataset:


![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/0e49ceac-f611-4680-aee9-3bb0380e1543)


Training the dataset:


![image](https://github.com/Revathi-Dayalan/Ex.No.1---Data-Preprocessing/assets/96000574/bd924bed-5d50-4177-aabf-7641fd697f81)


## RESULT
Thus the given data is been processed successfully.


