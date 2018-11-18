#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import Util as ut
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import math
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb

#Carregando dados de treino
dt = pd.read_csv("Data/train.csv")
dtest = pd.read_csv("Data/test.csv")

#Categorizando variáveis nominais
dt["bairro"] = ut.categorizar(dt["bairro"], "bairro", pd)
dt["tipo_vendedor"] = ut.categorizar(dt["tipo_vendedor"], "tipo_vendedor", pd)
dt["tipo"] = ut.categorizar(dt["tipo"], "tipo", pd)
dt['preco'] = dt['preco'].apply(np.log)
dt['area_util'] = dt['area_util'].apply(np.log)
#dt['bairro'] = dt['bairro'].apply(np.log)


dt = dt.drop(dt.columns[[0,8,9]], axis=1)

ut.corr_matrix(dt)




dtest["bairro"] = ut.categorizar(dtest["bairro"], "bairro", pd)
dtest["tipo_vendedor"] = ut.categorizar(dtest["tipo_vendedor"], "tipo_vendedor", pd)
dtest["tipo"] = ut.categorizar(dtest["tipo"], "tipo", pd)
dtest['area_util'] = dtest['area_util'].apply(np.log)
#dtest['bairro'] = dtest['bairro'].apply(np.log)


#dtest = dtest.drop(dtest.columns[[1,2,7,9,11,12,13,14,15]], axis=1)
dtest = dtest.drop(dtest.columns[[0,8,9]], axis=1)

dt = dt.drop(dt[(dt['preco'] > 17.5)].index)
dt = dt.drop(dt[(dt['preco'] < 7.5)].index)
ut.corr_matrix(dt)
# ut.histogram(dt)
# ut.boxplot(dt)

X = dt.iloc[:, :-1].values
Y = dt.iloc[:, -1].values

Xtest = dtest.iloc[:, :].values

regressor = LinearRegression(n_jobs=4, fit_intercept=True)

regressor.fit(X, Y)

# ytrain_pred = regressor.predict(X)
# ytest_pred = regressor.predict(Xtest)

# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring = scorer, cv = 10))
    return(rmse)

# print('\nDesempenho no conjunto de treinamento:')
# print('MSE  = %.5f' %           mean_squared_error(Y, ytrain_pred) )
# print('RMSE = %.5f' % math.sqrt(mean_squared_error(Y, ytrain_pred)))
# print('R2   = %.5f' %                     r2_score(Y, ytrain_pred) )
# print "RMSE on Training set :", rmse_cv_train(regressor).mean()







#0.5213
xgboost = xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.0468, learning_rate=0.04, max_depth=3, min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.9, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =5, n_jobs=5)
xgboost.fit(X, Y)

ytrain_pred = xgboost.predict(X)
ytest_pred = xgboost.predict(Xtest)

print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.5f' %           mean_squared_error(Y, ytrain_pred) )
print('RMSE = %.5f' % math.sqrt(mean_squared_error(Y, ytrain_pred)))
print('R2   = %.5f' %                     r2_score(Y, ytrain_pred) )
print "XGBOOST - RMSE on Training set :", rmse_cv_train(xgboost).mean()

v = np.vectorize(np.exp)
Y = v(Y)
ytrain_pred = v(ytrain_pred)
ytest_pred = v(ytest_pred)

file_ob = open("testSub.csv", "w")
file_ob.write("Id,preco\n")

for i in range(len(ytest_pred)):
    file_ob.write(str(i)+","+str(ytest_pred[i])+"\n")
file_ob.close()
