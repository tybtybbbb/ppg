import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


gamer=pd.read_csv("C:\\Users\\tybty\\Downloads\\gamer1.csv")
annotation = pd.read_csv("C:\\Users\\tybty\\Downloads\\gamer1-annotations.csv")
#annotation['Datatime'].apply(lambda(x:x.replace("T"," ")))
annotation['Datetime']=annotation['Datetime'].str.replace('T',' ')
#Stanford Sleepiness Self-Assessment (1-7)
annotation=annotation[annotation['Event']=='Stanford Sleepiness Self-Assessment (1-7)']
annotation['hour']=annotation['Datetime'].str[0:13]
annotation['minute']=annotation['Datetime'].str[0:16]
gamer['hour']=gamer['Time'].str[0:13]
df=gamer.merge(annotation, how='left',left_on='hour',right_on='hour')
columns = [ 'bpm', 'pnn20', 'pnn50', 'lf', 'hf', 'ratio']
X=df[columns]
y=df['Value']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)


#feature importance
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
for f in range(min(20,X_train.shape[1])):
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],  color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

#knn
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)
print(classification_report(y_test,knn_predictions))
confusion_matrix(y_test, knn_predictions)

# #logistic
# log_model = linear_model.LogisticRegression()
# log_model.fit(X = X_train, y = y_train)
# print(log_model.coef_)
# preds = log_model.predict_proba(X= X_train)
# predsdf = pd.DataFrame(preds)
# predsdf['result'] = [0 if x >.5 else 1 for x in predsdf[0]]
# accuracy_score(y_train, predsdf['result'])

#XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#SVM Classification Training
svm = SVC(C = .5)
svm.fit(X_train, y_train)
print(svm.score(X_train,y_train))
predictions = svm.predict(X_test)  # predict the target of testing samples
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)
print(gnb.score(X_train,y_train))
predictions = gnb.predict(X_test)  # predict the target of testing samples
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Artificial Neural Network
ann = MLPClassifier(learning_rate = 'adaptive')
ann.fit(X, y)
print(ann.score(X_train,y_train))
predictions = ann.predict(X_test)  # predict the target of testing samples
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#kfold xgboost
from sklearn.model_selection import cross_val_score
clf = xgb.XGBClassifier(
        learning_rate=0.009,
        max_depth=10,
        boosting='gbdt',
        objective='multi:softmax',
        metric='auc',
        seed=4,
        num_iterations=10000,
        early_stopping_round=100,
        verbose_eval=200,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )
scores = cross_val_score(clf, X, y, cv=5)
np.mean(scores)