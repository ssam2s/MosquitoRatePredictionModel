from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dataset Preprocessing

df = pd.read_csv("./mos_train.csv")

train, test = train_test_split(df, test_size=0.15, random_state=1)
train, val = train_test_split(train, test_size=0.18, random_state=1)

# Standard Model

majority_class = train['모기라벨'].mode()[0]

y_pred = [majority_class] * len(val)

print("\n최빈 클래스: ", majority_class)
print("validation 데이터셋 정확도: ", accuracy_score(val["모기라벨"], y_pred))

feature = ["평균기온", "최저기온", "최고기온", "일강수량", "평균 증기압", "평균 전운량", "상대습도"]
target = "모기라벨"

# Feature Matrix and Target Vector

x_train = train[feature]
y_train = train[target]

x_val = val[feature]
y_val = val[target]

x_test = test[feature]
y_test = test[target]

print("\nFeature Matrix: ", x_train.shape, x_val.shape, x_test.shape)
print("Target Vector: ", y_train.shape, y_val.shape, y_test.shape)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Logistic Regression Model

logistic = LogisticRegression()
logistic.fit(x_train_scaled, y_train)

# Eventual Model Generation

x_total = pd.concat([x_train, x_test])
y_total = pd.concat([y_train, y_test])

scaler = StandardScaler()

x_total_scaled = scaler.fit_transform(x_total)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_total_scaled, y_total)

lr = model
test_X = x_test_scaled
test_Y = y_test

train_X = x_total_scaled
train_Y = y_total

print("\nTrain Score: ", lr.score(train_X, train_Y))
print("Accuracy: ", lr.score(test_X, test_Y))

logit_roc_auc = roc_auc_score(test_Y, lr.predict(test_X))
fpr, tpr, thresholds = roc_curve(test_Y, lr.predict_proba(test_X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
