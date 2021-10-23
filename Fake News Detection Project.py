import pandas as pd 
import numpy as np


data = pd.read_csv(r'C:\Users\Raj\Downloads\news\news.csv')
data.head(5)


data=data.drop(['Unnamed: 0'], axis=1)
data=data[0:1000]


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X[0]
y[0]


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
mat_body = cv.fit_transform(X[:,1]).todense()
mat_body 


cv_head = CountVectorizer(max_features=5000)
mat_head = cv_head.fit_transform(X[:,0]).todense()
X_mat = np.hstack(( mat_head, mat_body))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_mat,y, test_size=0.2, random_state=0)


from sklearn.tree import DecisionTreeClassifier 
dtc = DecisionTreeClassifier(criterion= 'entropy')
dtc.fit(X_train, y_train)
dtc.predict(X_test)


from sklearn.tree import DecisionTreeClassifier 
dtc = DecisionTreeClassifier(criterion= 'entropy')
dtc.fit(X_train, y_train)


Y_pred = dtc.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, Y_pred)
