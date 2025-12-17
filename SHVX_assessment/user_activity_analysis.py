import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load dataset (LOCAL)
df = pd.read_csv("Social Media Users.csv")

print(df.head())
print(df.info())


# Convert Date Joined to datetime
df['Date Joined'] = pd.to_datetime(df['Date Joined'])

# Feature Engineering: Account age (in days)
df['Account_Age_Days'] = (pd.Timestamp.today() - df['Date Joined']).dt.days

# Encode target variable
df['Verified Account'] = df['Verified Account'].map({'Yes': 1, 'No': 0})

# Drop original date column
df.drop(columns=['Date Joined'], inplace=True)

# Check missing values
print(df.isnull().sum())

# Fill numeric missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)


categorical_cols = ['Platform', 'Owner', 'Primary Usage', 'Country']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


sns.countplot(x='Verified Account', data=df)
plt.title("Verified vs Non-Verified Users")
plt.show()


sns.boxplot(x='Verified Account', y='Daily Time Spent (min)', data=df)
plt.title("Daily Time Spent vs Verification Status")
plt.show()


sns.boxplot(x='Verified Account', y='Account_Age_Days', data=df)
plt.title("Account Age vs Verification")
plt.show()


#Corelation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['Daily Time Spent (min)', 'Account_Age_Days', 'Verified Account']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#test-train split
X = df.drop('Verified Account', axis=1)
y = df['Verified Account']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)


#Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


#Model Evaluation
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))


