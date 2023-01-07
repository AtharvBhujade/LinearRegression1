import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("Book1.csv")

data = data[(data.iloc[:, 1:] != 0).all(1)]

predict = "C"

data = data[["A", "B", "C"]]
data = shuffle(data)

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("bestmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open("bestmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print("Best accuracy :", best)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")
predicted = linear.predict(x_test)

for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

plt.scatter(data["A"], data["C"], color="red")
plt.scatter(data["B"], data["C"], color="blue")
plt.legend(loc=4)
plt.xlabel("A & B")
plt.ylabel("C")
plt.show()
