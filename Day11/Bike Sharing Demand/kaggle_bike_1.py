import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os 
os.chdir(r"C:\Training\Kaggle\Competitions\Bike Sharing Demand")

train = pd.read_csv("train.csv", parse_dates=['datetime'])
test = pd.read_csv("test.csv", parse_dates=['datetime'])
submit = pd.read_csv("sampleSubmission.csv")

train['day'] = train['datetime'].dt.day
train['month'] = train['datetime'].dt.month 
train['year'] = train['datetime'].dt.year 
train['hour'] = train['datetime'].dt.hour
train['wday'] = train['datetime'].dt.weekday


test['day'] = test['datetime'].dt.day
test['month'] = test['datetime'].dt.month 
test['year'] = test['datetime'].dt.year 
test['hour'] = test['datetime'].dt.hour
test['wday'] = test['datetime'].dt.weekday

X_train = train.drop(['datetime', 'count',
                      'casual', 'registered'],axis=1)
y_train = train['count']
X_test = test.drop('datetime', axis=1)

X_train['season'] = X_train['season'].astype(str)
X_train['weather'] = X_train['weather'].astype(str)
X_test['season'] = X_test['season'].astype(str)
X_test['weather'] = X_test['weather'].astype(str)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred[y_pred < 0] = 0

submit['count']= y_pred
submit.to_csv('sbt_1Jul2023.csv', index=False)
