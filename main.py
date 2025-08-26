import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

import joblib


df = pd.read_csv("covertype.csv")

#print(df.shape)

# Data Clean / Preprocessing

df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'], axis=1, inplace=True)

# Split & Scale Data Set

X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_f1 = 0
solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky"]
best_report = {}

# Logistic Regression Model Building
for x in range(0, 4): # I forgot to add a try-except wrapper but definetely do so
    lr = LogisticRegression(random_state=42, class_weight='balanced', solver=solvers[x], max_iter=2000)
    
    print("Beginning testing")

    lr.fit(X_train, y_train)
    log_y_pred = lr.predict(X_test) 
    report = classification_report(y_test, log_y_pred, output_dict=True)

    print("Logistic Regression F1-Score: ", report['weighted avg']['f1-score'])

    if report['weighted avg']['f1-score'] > best_f1:
        print("best solver: ", solvers[x])

        best_f1 = report['weighted avg']['f1-score']
        best_report = classification_report(y_test, log_y_pred)
        matrix = confusion_matrix(y_test, log_y_pred)

        joblib.dump(lr, 'logistic_regression_model.joblib')

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Logistic Regression (Solver: {solvers[x]})')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    elif report['weighted avg']['f1-score'] == best_f1:
        print("equal score as best: ", solvers[x])
        #wont save them, but we could if we wanted

    else:
        print("worse solver: ", solvers[x])

print("\n\n All Test Ended \n\n")
print(best_report)
print("---" * 20)

# Random Forest Model Building

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=1)
rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)

print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print("\n\n Class Reprort: ")
print(classification_report(y_test, rf_y_pred))

X_og = df.drop(columns=['Cover_Type'])
feature_names = X_og.columns

sig_features = pd.Series(rf.feature_importances_, index=feature_names)

top_features = sig_features.nlargest(20)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_features, y=top_features.index)

plt.title('Top 20 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()