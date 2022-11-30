import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm

# Loading the datasets into a dataframe
dataset = pd.read_csv('b_datasets.csv')

X = dataset.iloc[:, 0: 10].values
Y = dataset.iloc[:, -1].values

# Divinding the datasets into traning and test subsets
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.3, random_state=42)

# Scaling the data
sc = StandardScaler()
sc.fit(train_X, test_X)
train_X_std = sc.transform(train_X)
test_X_std = sc.transform(test_X)

####################################


start_time = time.time() # Generate starting timestamp

nb_model = GaussianNB()
# Classifier is trained with traning set
nb_model.fit(train_X_std, train_Y)

# Predicting testing set
predic_set_Y = nb_model.predict(test_X_std)

# Generate finishing timestamp

finish_time = time.time()

# Creating confusion matrix and performance metrices
cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y)
rec = recall_score(test_Y, predic_set_Y)
f_mea = f1_score(test_Y, predic_set_Y)
process_time = finish_time - start_time 


# Outouting the confusion matrix

tic_labels = ["Benign", "Malicious"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Naive Bayes Classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()


# Printing the perfomance metrics

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#############################################################

# Binary Classification - Random Forest

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators= 18)
rf_model.fit(train_X_std,train_Y)

predic_set_Y =rf_model.predict(test_X_std)

finish_time = time.time()

cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y)
rec = recall_score(test_Y, predic_set_Y)
f_mea = f1_score(test_Y, predic_set_Y)
process_time = finish_time - start_time

tic_labels = ["Benign", "Malicious"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for Random Forest classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#############################################################

# K Nearset Neighbours - Binary Classification

start_time = time.time() # Generate starting timestamp

kn_model = KNeighborsClassifier(n_neighbors= 4)

kn_model.fit(train_X_std, train_Y)

# Predicting testing set
predic_set_Y = kn_model.predict(test_X_std)

# Generate finishing timestamp

finish_time = time.time()

# Creating confusion matrix and performance metrices
cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y)
rec = recall_score(test_Y, predic_set_Y)
f_mea = f1_score(test_Y, predic_set_Y)
process_time = finish_time - start_time 


# Outouting the confusion matrix

tic_labels = ["Benign", "Malicious"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for KNN classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()


# Printing the perfomance metrics

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#############################################################

# Support Vector - Binary Classification 

start_time = time.time() # Generate starting timestamp

svm_model = svm.SVC(kernel='poly')

svm_model.fit(train_X_std, train_Y)

# Predicting testing set
predic_set_Y = svm_model.predict(test_X_std)

# Generate finishing timestamp

finish_time = time.time()

# Creating confusion matrix and performance metrices
cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y)
rec = recall_score(test_Y, predic_set_Y)
f_mea = f1_score(test_Y, predic_set_Y)
process_time = finish_time - start_time 


# Outouting the confusion matrix

tic_labels = ["Benign", "Malicious"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Confusion Matrix for SVM classifier');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()


# Printing the perfomance metrics

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#############################################################
# Changing datasets to only malicious traffic

# Loading the datasets into a dataframe
dataset = pd.read_csv('./m_data.csv')

X = dataset.iloc[:, 0: 10].values
Y = dataset.iloc[:, -1].values

# Divinding the datasets into traning and test subsets
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.3, random_state=42)

# Scaling the data
sc = StandardScaler()
sc.fit(train_X, test_X)
train_X_std = sc.transform(train_X)
test_X_std = sc.transform(test_X)

####################################
# Naive Bayes - Multiclass Classification

start_time = time.time()


nb_model = MultinomialNB()
nb_model.fit(train_X, train_Y)

predic_set_Y = nb_model.predict(test_X)

finish_time = time.time()

cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y, average= 'weighted')
rec = recall_score(test_Y, predic_set_Y, average= 'weighted')
f_mea = f1_score(test_Y, predic_set_Y, average= 'weighted')
process_time = finish_time - start_time

tic_labels = ["Benign", "Malicious","Miner", "Ransom"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Naive Bayes - Multiclass Classification');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#######################################################
# Random Forest - Multiclass Classification

start_time = time.time()


rf_model = RandomForestClassifier(n_estimators = 13)
rf_model.fit(train_X_std, train_Y)

predic_set_Y = rf_model.predict(test_X_std)

finish_time = time.time()

cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y, average= 'weighted')
rec = recall_score(test_Y, predic_set_Y, average= 'weighted')
f_mea = f1_score(test_Y, predic_set_Y, average= 'weighted')
process_time = finish_time - start_time

tic_labels = ["Benign", "Malicious","Miner", "Ransom"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('Random Forest - Multiclass Classification');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#######################################################

# k Nearest Neighbours - Multiclass Classification

start_time = time.time()


kn_model = KNeighborsClassifier(n_estimators = 4)
kn_model.fit(train_X_std, train_Y)

predic_set_Y = kn_model.predict(test_X_std)

finish_time = time.time()

cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y, average= 'weighted')
rec = recall_score(test_Y, predic_set_Y, average= 'weighted')
f_mea = f1_score(test_Y, predic_set_Y, average= 'weighted')
process_time = finish_time - start_time

tic_labels = ["Benign", "Malicious","Miner", "Ransom"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('k Nearest Neighbours - Multiclass Classification');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")
#######################################################

# Support Vector Machines - Multiclass Classification
start_time = time.time()


svm_model = svm.SVC(kernel='poly')
svm_model.fit(train_X_std, train_Y)

predic_set_Y = svm_model.predict(test_X_std)

finish_time = time.time()

cm = confusion_matrix(test_Y, predic_set_Y)
acc = accuracy_score(test_Y, predic_set_Y)
pre = precision_score(test_Y, predic_set_Y, average= 'weighted')
rec = recall_score(test_Y, predic_set_Y, average= 'weighted')
f_mea = f1_score(test_Y, predic_set_Y, average= 'weighted')
process_time = finish_time - start_time

tic_labels = ["Benign", "Malicious","Miner", "Ransom"]
ax = sns.heatmap(cm / np.sum(cm), annot= True, fmt='.2%',
                 xticklabels= tic_labels, yticklabels= tic_labels, cmap = 'Blues')
ax.set_title('k Nearest Neighbours - Multiclass Classification');
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()

print("Accuracy Score: " + str(acc * 100)+ "%")
print("precision Score: " + str(pre * 100)+ "%")
print("Recall Score: " + str(rec * 100)+ "%")
print("F_measure Score: " + str(f_mea * 100)+ "%")
print("Processing time: " + str(process_time) + "seconds")