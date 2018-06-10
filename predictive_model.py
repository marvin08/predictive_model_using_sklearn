# This Predictive Model is built using sklearn.
# Author : Akshay Shrimali, Github : @marvin08

from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

filename = 'BBC.csv'
dataframe = read_csv(filename)

# dataframe.shape -> This function returns the number of rows & columns.
# dataframe.describe() -> This function returns the mean, mode, median and other quantities of every column/feature.
# print(dataframe.groupby('BikeBuyer').size()) -> This function groups the data with a particular value of 'BikeBuyer' attribute together.

array = dataframe.values

X = array[:, 0:11]
Y = array[:, 11]

# 75:25 ratio is maintained for training & testing respectively.
test_size = 0.25
seed = 5

# Now, we split the dataset.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)

# Apart from SVC, other classifiers can also be used. Here, SVC is used as it gives the best accuracy.
model = SVC()

# Now, we train the model.
model.fit(X_train, Y_train)

# Accuracy is the best way to evaluate the model.
result = model.score(X_test, Y_test)
print("Accuracy: %.3f%%") % (result*100.0)
