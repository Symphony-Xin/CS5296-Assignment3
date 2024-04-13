from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

# Load data
train_dataset = pd.read_csv('./mnist_train.csv')
test_dataset = pd.read_csv('./mnist_test.csv')

X_train = train_dataset.drop('label',axis=1).values
y_train = train_dataset['label'].values
X_test = test_dataset.drop('label',axis=1).values
y_test = test_dataset['label'].values

# Scale data to improve SVM training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier
clf = svm.SVC(gamma=0.001)

# Train the classifier
start = time.time()
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted_train = clf.predict(X_train)
predicted_test = clf.predict(X_test)

# Print report
end = time.time()
print(f"Classification report for train {clf}:\n{metrics.classification_report(y_train, predicted_train)}\n")
print(f"Classification report for classifier {clf}:\n{metrics.classification_report(y_test, predicted_test)}")
with open('./SVM.txt','w') as file:
    file.write(f"Classification report for train {clf}:\n{metrics.classification_report(y_train, predicted_train)}\n")
    file.write(f"Classification report for test {clf}:\n{metrics.classification_report(y_test, predicted_test)}\nRuntime:{end - start}")
file.close()