def my_function():
  print("Hello from a function")
  
my_function()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv('./data/iris.csv')
iris.drop('id', axis=1, inplace=True)
print(iris.head())

# build a dict mapping species to an integer code
inv_name_dict = {'iris-setosa': 0, 'iris-versicolor': 1, 'iris-virginica': 2}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]
# scatter plot
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
## add legend
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
#plt.savefig("./figures/scattersepal.png")
#plt.show()

# Univariate Plot
iris.hist()
#plt.savefig("./figures/hist.png")
#plt.show()

scatter = plt.scatter(iris['petal_len'], iris['petal_wd'], c=colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
# add legend
plt.legend(handles= scatter.legend_elements()[0], labels = inv_name_dict.keys())
#plt.savefig("./figures/scatterpetal.png")
#plt.show()

# pd.plotting.scatter_matrix(iris, alpha=0.2, figsize=(6, 6), diagonal="kde")
pd.plotting.scatter_matrix(iris)
#plt.show()

X = iris[['petal_len', 'petal_wd']]
y = iris['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=1, stratify=y)
print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.neighbors import KNeighborsClassifier
## instantiate
knn = KNeighborsClassifier(n_neighbors=5)
## fit
print(knn.fit(X_train, y_train))

# predict
y_pred = knn.predict(X_test)
print(y_pred[:5])

# probabilities prediction
y_pred_prob = knn.predict_proba(X_test)
print(y_pred[10:12])
print(y_pred_prob[10:12])

# Measure Accuracy
print((y_pred==y_test.values).sum())
print(y_test.size)

# Accuracy
print((y_pred==y_test.values).sum()/y_test.size)
# Accuracy using knn.score method
print(knn.score(X_test, y_test))
# Accuracy using accuracy_score method
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig("./figures/confusionmatrix.png")
plt.show()

