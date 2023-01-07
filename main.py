import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("Book1.csv")

data = data[["A", "B", "C"]]

data = data[(data.iloc[:, 1:] != 0).all(1)]

# data.to_excel("flitered.xlsx", index=None, header=True)

predict = "C"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
# print(acc)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
best = 0
n = int(input("Enter number of try's :"))
for _ in range(n):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("model4.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Best Accuracy :", best)
