from sklearn import metrics
from keras.models import Model
import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Dense
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import plot_confusion_matrix

data0 = pd.read_csv(
    r'C:\Users\Public\Documents\Phishing-Website-Detection\DataFiles\5.urldata.csv')
data0.head()


data0.columns
data0.info()
data0.describe()

# Plotting the data distribution
data0.hist(bins=50, figsize=(15, 20))
plt.show()

# plotting the correlation matrix
plt.figure(figsize=(15, 13))
sns.heatmap(data0.corr())
plt.show()

data = data0.drop(['Domain'], axis=1).copy()
# checking the data for null or missing values
data.isnull().sum()
data = data.sample(frac=1).reset_index(drop=True)
data.head()
# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label', axis=1)
X.shape, y.shape

# training the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12)
X_train.shape, X_test.shape


# mutual info cannot be negetive
mutual_info = mutual_info_classif(X_train, y_train)
# measures the dependency between the variables
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)
print(mutual_info)

mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


ML_Model = []
acc_train = []
acc_test = []


# function storing the result obtained after deploying every model
def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))


# # MODELS
# # 4 Different models are used below

# ## 1.SVM MODEL
# -------------------------------------------------------------------------
# Support vector machine model
# instantiate the model
svm = SVC(kernel='rbf', degree=6, C=1.0, random_state=12)
# fit the model
svm.fit(X_train, y_train)

y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)

# computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train, y_train_svm)
acc_test_svm = accuracy_score(y_test, y_test_svm)

print("SVM :- Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM :- Accuracy on test Data: {:.3f}".format(acc_test_svm))
storeResults('SVM', acc_train_svm, acc_test_svm)


# ### Confusoin Matrix
# Using linear regression----
clf = LogisticRegression()  # fit model
clf.fit(X_train, y_train)
disp = plot_confusion_matrix(clf, X_test, y_test)


# ## RANDOM FOREST MODEL
# ----------------------------------------------------------------------------
# instantiate the model
forest = RandomForestClassifier(max_depth=5)

# fit the model
forest.fit(X_train, y_train)
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

# computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train, y_train_forest)
acc_test_forest = accuracy_score(y_test, y_test_forest)

print("Random forest:- Accuracy on training Data: {:.3f}".format(
    acc_train_forest))
print("Random forest:- Accuracy on test Data: {:.3f}".format(acc_test_forest))
storeResults('Random Forest', acc_train_forest, acc_test_forest)


plt.figure(figsize=(9, 7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


# ## XGBoost CLASSIFICATION MODEL
# #---------------------------------------------------------------------------
#
# instantiate the model
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
# fit the model
xgb.fit(X_train, y_train)

# predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

# computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train, y_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_test_xgb)

print("XGBoost :- Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost :- Accuracy on test Data: {:.3f}".format(acc_test_xgb))

# again storing the output of accuracies
storeResults('XGBoost', acc_train_xgb, acc_test_xgb)


# Autoencoder Model
# ------------------------------------------------------------------------------
input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)

decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()


# compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Training the model
history = autoencoder.fit(X_train, X_train, epochs=10,
                          batch_size=64, shuffle=True, validation_split=0.2)


acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder:- Accuracy on training Data: {:.3f}' .format(
    acc_train_auto))
print('Autoencoder:- Accuracy on test Data: {:.3f}' .format(acc_test_auto))


# storing the results. Contains accuracy output of every model used above
# Note: Execute only once to avoid duplications.
storeResults('AutoEncoder', acc_train_auto, acc_test_auto)
# creating dataframe
results = pd.DataFrame({'ML Model': ML_Model,
                        'Train Accuracy': acc_train,
                        'Test Accuracy': acc_test})

print(results)
