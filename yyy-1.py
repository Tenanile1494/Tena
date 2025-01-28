# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Step 2: Load Sample Data
data = {
    'PassengerId': [1, 2, 3, 4, 5],
    'Survived': [0, 1, 1, 0, 1],
    'Pclass': [3, 1, 2, 1, 3],
    'Sex': ['male', 'female', 'female', 'female', 'male'],
    'Age': [22, 38, 26, 35, 35],
    'SibSp': [1, 1, 0, 1, 0],
    'Parch': [0, 0, 0, 0, 0],
    'Fare': [7.25, 71.83, 8.05, 53.10, 8.05],
    'Embarked': ['S', 'C', 'S', 'S', 'S']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Step 3: Data Cleaning and Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing ages
df.dropna(subset=['Embarked'], inplace=True)  # Drop rows with missing 'Embarked'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Binary encoding for 'Sex'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # One-hot encoding

# Select relevant features
features = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q']]
target = df['Survived']

# Step 4: Train Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)

# Step 5: Visualization
# Age distribution of survivors and non-survivors
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Survived'] == 1]['Age'], bins=30, color='blue', label='Survived', kde=True)
sns.histplot(df[df['Survived'] == 0]['Age'], bins=30, color='red', label='Not Survived', kde=True)
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Count plot for passenger class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

print(f"ROC AUC Score: {roc_auc:.2f}")
