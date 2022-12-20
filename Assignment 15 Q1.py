# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:11:17 2022

@author: SANJUSHA
"""

# RANDOM FOREST CLASSIFIER

import pandas as pd
import numpy as np

df=pd.read_csv("Fraud_check.csv")
df
df.isnull().sum()
df.info()

# Boxplots
df.boxplot("City.Population",vert=False)
df.boxplot("Work.Experience",vert=False)
# There are no outliers

df["Taxable.Income"]=pd.cut(df["Taxable.Income"], bins=[0,30000,99620],labels=["Risky","Good"])
df

# Splitting the variables
Y=df["Taxable.Income"]

X1=df.iloc[:,:2]
X2=df.iloc[:,3:]
X=pd.concat([X1,X2],axis=1)
X.dtypes

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["Undergrad"]=LE.fit_transform(X["Undergrad"])
X["Undergrad"]=pd.DataFrame(X["Undergrad"])

X["Marital.Status"]=LE.fit_transform(X["Marital.Status"])
X["Marital.Status"]=pd.DataFrame(X["Marital.Status"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["City.Population"]=MM.fit_transform(X[["City.Population"]])
X["Work.Experience"]=MM.fit_transform(X[["Work.Experience"]])
X

Y=LE.fit_transform(df["Taxable.Income"])
Y=pd.DataFrame(Y)

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Model fitting
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_depth=5,max_leaf_nodes=25)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
ac2=accuracy_score(Y_test,Y_predtest)

# If max_depth is 5, max_leaf_nodes is 25 and test_size=0.2 then ac1=80% and ac2=78%
# If max_depth is 6, max_leaf_nodes is 15 and test_size=0.3 then ac1=89% and ac2=75%

from sklearn.model_selection import GridSearchCV
d1={'max_depth':np.arange(0,50,1),
     'max_leaf_nodes':np.arange(0,50,1)}

Gridgb=GridSearchCV(estimator=RandomForestClassifier(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_


#===================================================================================================#


# RANDOM FOREST REGRESSOR

import pandas as pd
import numpy as np

df=pd.read_csv("Fraud_check.csv")
df
df.isnull().sum()
df.info()

Y=df["Taxable.Income"]

X1=df.iloc[:,:2]
X2=df.iloc[:,3:]
X=pd.concat([X1,X2],axis=1)
X.dtypes

# Standardization

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["Undergrad"]=LE.fit_transform(X["Undergrad"])
X["Undergrad"]=pd.DataFrame(X["Undergrad"])

X["Marital.Status"]=LE.fit_transform(X["Marital.Status"])
X["Marital.Status"]=pd.DataFrame(X["Marital.Status"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["City.Population"]=MM.fit_transform(X[["City.Population"]])
X["Work.Experience"]=MM.fit_transform(X[["Work.Experience"]])
X

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Model fitting
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=5,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import r2_score
rs1=r2_score(Y_train,Y_predtrain) # 0.82304
rs2=r2_score(Y_test,Y_predtest) # 0.78561


from sklearn.model_selection import GridSearchCV
d1={'max_depth':np.arange(0,50,1),
     'max_leaf_nodes':np.arange(0,50,1)}

Gridgb=GridSearchCV(estimator=RandomForestRegressor(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_



