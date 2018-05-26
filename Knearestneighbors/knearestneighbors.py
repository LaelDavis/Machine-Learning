import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

#import data from .txt file
df = pd.read_csv('breast-cancer.data.txt', error_bad_lines=False)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class']) #default axis=0

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#define the classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
#address the 'deprication' warning
example_measures = example_measures.reshape(len(example_measures), -1)


prediction = clf.predict(example_measures)
print(prediction)