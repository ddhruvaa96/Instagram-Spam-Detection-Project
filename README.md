import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


# -----------------------------
# Load Dataset
# -----------------------------

proj_path = os.getcwd()
print("Project Path =", proj_path)

file_path = proj_path + "\\project_data.xlsx"
print("File Path =", file_path)

df = pd.read_excel(file_path)

print(df)

# -----------------------------
# Dataset Information
# -----------------------------

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows =", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nStatistical Summary")
print(df.describe())

print("\nFake / Genuine Count")
print(df['is_fake'].value_counts())

# -----------------------------
# Graphs
# -----------------------------

# Bar Graph
df['is_fake'].value_counts().plot(kind='bar')
plt.title("Fake vs Genuine Accounts")
plt.xlabel("0 = Genuine     1 = Fake")
plt.ylabel("Number of Accounts")
plt.grid()
plt.show()

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(df['edge_followed_by'],
            df['edge_follow'],
            c=df['is_fake'])
plt.xlabel("Followers")
plt.ylabel("Following")
plt.title("Followers vs Following")
plt.show()

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['username_length'], bins=15)
plt.title("Username Length Distribution")
plt.xlabel("Username Length")
plt.ylabel("Frequency")

plt.show()

# Box Plot
plt.figure(figsize=(6,5))
plt.boxplot(df['edge_followed_by'])
plt.title("Followers Distribution")
plt.show()

# Pie Chart
df['is_fake'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.ylabel("")
plt.title("Fake vs Genuine Accounts")

plt.show()


# -----------------------------
# Input and Target
# -----------------------------

inp = df.drop('is_fake', axis=1)

target = df['is_fake']

print("\nInput")
print(inp)

print("\nTarget")
print(target)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    inp,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# =====================================================
# KNN
# =====================================================

print("\n========== KNN ==========")

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("Accuracy =", accuracy_score(y_test, knn_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, knn_pred))

print("\nClassification Report")
print(classification_report(y_test, knn_pred))

# -----------------------------
# Confusion Matrix Graph
# -----------------------------

ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.title("KNN Confusion Matrix")
plt.show()

# =====================================================
# Decision Tree
# =====================================================

print("\n========== Decision Tree ==========")

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Accuracy =", accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))


plt.figure(figsize=(20,10))
plot_tree(
    dt,
    feature_names=inp.columns,
    class_names=["Genuine", "Fake"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()

# -----------------------------
# Confusion Matrix Graph
# -----------------------------

ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# =====================================================
# Test New Instagram Account
# =====================================================

d = {
    'edge_followed_by': 0.002,
    'edge_follow': 0.150,
    'username_length': 12,
    'username_has_number': 1,
    'full_name_has_number': 0,
    'full_name_length': 15,
    'is_private': 1,
    'is_joined_recently': 0,
    'has_channel': 0,
    'is_business_account': 0,
    'has_guides': 0,
    'has_external_url': 0
}

test_inp = pd.DataFrame([d])

print("\nTest Input")
print(test_inp)

knn_result = knn.predict(test_inp)

dt_result = dt.predict(test_inp)

print("\nKNN Prediction")

if knn_result[0] == 1:
    print("Fake Account")
else:
    print("Genuine Account")

print("\nDecision Tree Prediction")

if dt_result[0] == 1:
    print("Fake Account")
else:
    print("Genuine Account")
    
print("\n==============================")
print("Model Comparison")
print("==============================")

print("KNN Accuracy           :", round(accuracy_score(y_test, knn_pred) * 100, 2), "%")
print("Decision Tree Accuracy :", round(accuracy_score(y_test, dt_pred) * 100, 2), "%")
 
