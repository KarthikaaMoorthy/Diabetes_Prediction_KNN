#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Load the dataset
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\diabetes_ML_Dataset.csv")

#replace zeros
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

#split the dataset into train and test data
X = dataset.iloc[:, 0:8]
Y = dataset.iloc[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Define the model
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

#fit model
classifier.fit(X_train, Y_train)

#predict the test set results
Y_pred = classifier.predict(X_test)

#Evaluate model
cm = confusion_matrix(Y_test, Y_pred)
print("Performance of the diabetics model of the patient",cm)
print("f1 score",f1_score(Y_test, Y_pred))

#accuracy
print("Accuracy score of the patient being suffered from diabetes is",accuracy_score(Y_test, Y_pred))
