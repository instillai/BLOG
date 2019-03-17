import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Determine the column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'categories']
iris = pd.read_csv(url, header=None, names=col_names)

# ### PLOT ###
# category_1 = iris[iris['categories']=='Iris-setosa']
# category_2 = iris[iris['categories']=='Iris-virginica']
# category_3 = iris[iris['categories']=='Iris-versicolor']
#
# fig, ax = plt.subplots()
# ax.plot(category_1['sepal_length'], category_1['sepal_width'], marker='o', linestyle='', ms=12, label='Iris-setosa')
# ax.plot(category_2['sepal_length'], category_2['sepal_width'], marker='o', linestyle='', ms=12, label='Iris-virginica')
# ax.plot(category_3['sepal_length'], category_3['sepal_width'], marker='o', linestyle='', ms=12, label='Iris-versicolor')
# ax.legend()
# plt.show()

# Creating the dictionary of categories and forming the labels vector.
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['labels'] = [iris_class[i] for i in iris.categories]

# Creating the data and label vectors. The iris.drop eliminates the irrelevant columns.
X = iris.drop(['categories', 'labels'], axis=1)
Y = iris.labels

# Split the data into train/test.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)


from sklearn.neighbors import KNeighborsClassifier

## Call the model with k=10 neighbors.
knn = KNeighborsClassifier(n_neighbors=10)

## Fit the model using the training data.
knn.fit(X_train, Y_train)

## Test phase.
print(knn.score(X_test, Y_test))


