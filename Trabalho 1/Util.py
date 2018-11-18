#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tipo = {"Casa": 4, "Apartamento": 3, "Loft": 2, "Quitinete": 1}
tipo_vendedor = {"Imobiliaria": 1, "Pessoa Fisica": 2}
bairro = {}

mapeamento = {
    "tipo": tipo,
    "tipo_vendedor": tipo_vendedor,
    "bairro": {}
}


def categorizar(col, dict, pd):
    colCoded = pd.Series(col, copy=True)
    if dict == 'bairro':
        generateMap(col, dict)
    for key, value in mapeamento[dict].items():
        colCoded.replace(key, value, inplace=True)
    return colCoded


def generateMap(col, colName):
    index = 1
    for i in col:
        if i not in mapeamento[colName]:
            mapeamento[colName][i] = index
            index += 1

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()




def plot_correlation_matrix(df):
    import matplotlib.pyplot as plt
    """Takes a pandas dataframe as input"""
    fig, ax = plt.subplots(nrows=1, ncols=1)

    cax = ax.matshow(df.corr())

    ticks = list(range(len(df.columns)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(df.columns, rotation=20, horizontalalignment='left')
    ax.set_yticklabels(df.columns)

    plt.tight_layout()
    plt.show()


def corr_matrix(data):
    corrmat = data.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat,vmax=1,square=True)
    plt.savefig('corr.png')
    plt.cla()
    plt.clf()

def histogram(dt):
    for col in dt:
        histo = sns.distplot(dt[col])
        fig = histo.get_figure()
        fig.savefig('Histogramas/Histogram_'+col+'.png')
        #fig.cla()
        fig.clf()

# def scatter(col, dt):
#     data = pd.concat([dt['preco'], data_train[col]],axis=1)
#     data.plot.scatter(x=col,y='SalePrice', ylim=(0,800000)) # limite maximo baseado no describe
#     plt.savefig('human-research/Scatter_SalePrice_'+col+'.png')
#     plt.cla()
#     plt.clf()
#
def boxplot(dt):
    for col in dt:
        try:
            data = pd.concat([dt['preco'], dt[col]], axis=1)
            f, ax = plt.subplots(figsize=(8, 6))
            fig = sns.boxplot(x=col, y="preco", data=data)
            fig.axis(ymin=10)
            plt.savefig('Boxplot/BoxPlot_'+col+'.png')
            plt.cla()
            plt.clf()
        except ValueError:
            print col