import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


# Load Data
riskClass = pd.read_csv('train_riskClass.csv')
print('Shape of the Train Dataframe :', riskClass.shape)


# Make a copy of the data 
df = riskClass.copy()


# Convert Gender, Marital Status, & Mortgage to numeric features
df_cat = df[['Gender','MaritalStatus','Mortgage']]
colcat = df_cat.columns.tolist()
df_dum = pd.get_dummies( df_cat, prefix= colcat )
df = pd.concat( [riskClass, df_dum], axis=1)
df = df.drop(columns=colcat)
del riskClass, df_cat, colcat, df_dum 


# Convert Risk categories to numeric
target = 'Risk'
target_mapper = {'Good Risk': 0, 'Bad Risk': 1}


def target_converter(key):
    if type(key) != str:
        val = 0
    else:
        val = target_mapper[key]
    return val


df['Risk'] = df['Risk'].apply( target_converter )

del target_mapper, target_converter
print('Shape of the final dataframe :', df.shape)
#print(df.head(5))

# Separating X and y
X = df.drop('Risk', axis=1)
y = df['Risk']


# Build Random Forest model
clf = RandomForestClassifier()
clf.fit(X, y)


# Saving the model
pickle.dump(clf, open('model_riskClass.pkl', 'wb'))

# Clean Workspace
del df,X,y,clf