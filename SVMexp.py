from sklearn import datasets
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
import numpy as np
from SVM import SVM
from matplotlib import pyplot as plt

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X = normalize(X)
y = 2* y -1 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

mysvm = SVM(learning_rate = 1, iter = 4000, reg = 0.0001)
mysvm.fit(X_train, y_train)
y_pred = mysvm.predict(X_test)

acc = np.sum(y_pred == y_test)/len(y_test)
print(f"Accuracy = {acc}")
plt.plot(mysvm.Loss)
plt.show()


