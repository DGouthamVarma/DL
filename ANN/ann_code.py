#Importing our class
from ann import ArtificialNeuralNetwork

#Importing the required libraries for data preprocessing and metrics to evaluate model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#Data Preprocessing
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values
label_encoder = LabelEncoder()
X[:,1] = label_encoder.fit_transform(X[:,1])
X[:, 2] = label_encoder.fit_transform(X[:,2])
hot_encoder = OneHotEncoder(categorical_features=[1])
X = hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training the model
model = ArtificialNeuralNetwork(X_train, y_train, number_of_units = 6)
model.fit(number_of_epochs = 200, learning_rate = 0.01)

#Predicting the output values using the trained model
predicted_values = model.predict(X_test)
predicted_values = (predicted_values > 0.5)

#Evaluating the model using confusion matrix
cf = confusion_matrix(y_test, predicted_values)
print("Confusion matrix is ", cf)
