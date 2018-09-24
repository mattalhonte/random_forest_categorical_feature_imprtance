import pandas as pd
from toolz.functoolz import pipe
from toolz.dicttoolz import valmap
from functools import partial
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display



def getOneHotNames(col, df):
    vals = sorted(list(df[col].dropna().unique()))
    return [col + "Is" + str(val) for val in vals]

def varToBool(col, df):
    vals = sorted(list(df[col].dropna().unique()))
    
    dfToUse = df.copy()
    
    for val in vals:
        dfToUse[col + "Is" + str(val)] = (dfToUse[col]==val).map(int)
        
    return dfToUse.drop(columns=col)


def oneHotEncodeMultipleVars(varList, df):
    partialEncoders = (partial(varToBool, x) for x in varList)
    
    return pipe(df,
                *partialEncoders)

def getTotaledImportances(labels, forest):
    featDict = dict(zip(labels, forest.feature_importances_))
    uniqueLabels = set(x.split("Is")[0] for x in featDict.keys())
    
    featsAndOneHots =  {y: [x for x in labels if y in x] for y in uniqueLabels}
    
    summedFeats = valmap(lambda x: sum(featDict[y] for y in x), featsAndOneHots)
    
    return sorted(summedFeats.items(), key= lambda x: x[1], reverse=True)


def forestClassAccuracy(X, y, labels, metric, extraArgs={}):

    clf = RandomForestClassifier(random_state=0,
                                 **extraArgs)
    
    
    clf.fit(X, y)
    
    y_pred = clf.oob_decision_function_[:, 1]

    score = metric(y, y_pred)
    
    totaledImportance =  getTotaledImportances(labels, clf)
    
    
    return {
        "score": score,
        "features": labels,
        "model": clf,
        "featureImportances": pd.DataFrame(
            totaledImportance, 
            columns=[
                "Variable",
                "Importance",
            ]),
           }

def displayFeatureImportances(X, 
                              y, 
                              labels, 
                              metric, 
                              extraArgs={},
                              html=False):
    
    rfDct = forestClassAccuracy(X, 
                                y, 
                                labels, 
                                metric, 
                                extraArgs)
    
    print("Score is " + str(rfDct["score"]))
    
    df = rfDct["featureImportances"]
    
    if html==True:
        print(df
              .to_html())
        
    display(df
           )

