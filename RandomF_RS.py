import numpy as np
import pandas as pd

def transform_data(train_data_fname, test_data_fname):
    df_train = pd.read_csv(train_data_fname)
    df_train['is_train_set'] = 1
    df_test = pd.read_csv(test_data_fname)
    df_test['is_train_set'] = 0
    
    #descarto los valores NaN de DepartmentDescription
    df_train = df_train.dropna(subset = ['DepartmentDescription'])

    # we  get the TripType for the train set. To do that, we group by VisitNumber and
    # then we get the max (or min or avg)
    y = df_train.groupby(["VisitNumber", "Weekday"], as_index=False).max().TripType

    # we remove the TripType now, and concat training and testing data
    # the concat is done so that we have the same columns for both datasets
    # after one-hot encoding
    df_train = df_train.drop("TripType", axis=1)
    df = pd.concat([df_train, df_test])
    
    # the next three operations are the ones we have just presented in the previous lines
        
    # drop the columns we won't use (it may be good to use them somehow)
    df = df.drop(["FinelineNumber"], axis=1)

    # one-hot encoding for the DepartmentDescription
    df = pd.get_dummies(df, columns=["DepartmentDescription"], dummy_na=True)

    # now we add the groupby values
    df = df.groupby(["VisitNumber", "Weekday"], as_index=False).sum()
    
    # finally, we do one-hot encoding for the Weekday
    df = pd.get_dummies(df, columns=["Weekday"], dummy_na=True)

    # get train and test back
    df_train = df[df.is_train_set != 0]
    df_test = df[df.is_train_set == 0]
    
    X = df_train.drop(["is_train_set"], axis=1)
    yy = None
    XX = df_test.drop(["is_train_set"], axis=1)

    return X, y, XX, yy

X, y, XX, yy = transform_data("../data/train.csv", "../data/test.csv")


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

param_grid = {
    'n_estimators' : [50, 100, 150, 200, 250],
    'criterion' : ['gini', 'entropy'],
    'min_samples_split' : np.arange(2,6),
    'min_samples_leaf' : np.arange(1,6)
}

import warnings
warnings.filterwarnings('ignore')

model = RandomForestClassifier(random_state = 0)

cv = GridSearchCV(model, param_grid, n_jobs = -1, scoring='accuracy', cv=5)

cv.fit(X_train, y_train);  


results = cv.cv_results_
df = pd.DataFrame(results)
df.to_csv('resultados-RF.csv')

best_model = cv.best_estimator_

y_train_pred = best_model.predict(X_train)

report = classification_report(y_train, y_train_pred, output_dict=True)

df = pd.DataFrame(report).transpose()

df.to_csv('report_train-RF.csv')

y_valid_pred = best_model.predict(X_valid)

report = classification_report(y_valid, y_valid_pred, output_dict=True)

df = pd.DataFrame(report).transpose()

df.to_csv('report_valid-RF.csv')


from numpy import savetxt

savetxt('matrix_train-RF.csv', confusion_matrix(y_train, y_train_pred), delimiter=',')

y_valid_pred = best_model.predict(X_valid)


savetxt('matrix_train-RF.csv', confusion_matrix(y_valid, y_valid_pred), delimiter=',')

yy = best_model.predict(XX)

submission = pd.DataFrame(list(zip(XX.VisitNumber, yy)), columns=["VisitNumber", "TripType"])
submission.to_csv("../data/submission.csv", header=True, index=False)
