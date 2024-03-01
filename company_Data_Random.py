# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:30:04 2024

@author: Acer
"""

#Random Forest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("C:/12-Decision_Tree/Company_Data.csv")
dir(data)

data.head()
data.isnull().sum()
data.dropna()
#data.dropna():- removes the rows that contains null values

data.columns

data['Sales']=data.Sales

X=data.drop('Sales',axis='columns')
y=data.Sales

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

#n_estimators is the number of trees in the forest

model.fit(X_train,y_train)

model.score(X_test,y_test)

y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')