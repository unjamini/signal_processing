from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


path_to_train_dset = './train_df.csv'
path_to_test_dset = './test_df.csv'

train_df = pd.read_csv(path_to_train_dset)
test_df = pd.read_csv(path_to_test_dset)

X_train = train_df.drop(columns=['target'])
y_train = np.ravel(train_df['target'])

X_test = test_df.drop(columns=['target'])
y_test = np.ravel(test_df['target'])

tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
tree_clf = tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
