from sklearn import datasets
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DT


data = datasets.load_breast_cancer()
X, y = data.data, data.target
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

myDT = DT(min_Node_split = 1, max_depth = 100)
myDT.fit(X_train, y_train)
y_pred = myDT.predict(X_test)
acc = np.sum(y_pred == y_test)/len(y_test)
print(f"Accuracy = {acc}")