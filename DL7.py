# customer churn prediction using ANN : (customer churn means how many people are leaving the business)
# tenure in the columns' means how many years or months this customer is with service , how loyal customer is
# churn mean the customers are leaving or not


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

df = pd.read_csv('.\\datasets\\customer_churn.csv')

# cleaning our data:

df.drop('customerID', axis='columns', inplace=True)
df.TotalCharges.values  # our total charges is object because its strings
pd.to_numeric(df.TotalCharges, errors='coerce').isnull()  # converting strings into numeric ignoring errors and
# returning a list containing False and True , where value is null or not
df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]  # showing rows where total charges is null after
# numeric conversion

df1 = df[df.TotalCharges != ' ']  # dropping those null rows
df1.dtypes  # we can see our total charges value type is still object, and it has to be float type
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)  # now we do the numeric conversion

df1[df1.Churn == 'No']  # showing customers that aren't leaving

tenure_churn_no = df1[df1.Churn == 'No'].tenure  # saving dataframe where churn is no
tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure  # saving dataframe where churn is no

plt.xlabel('tenure')  # visualization of churn based on tenure
plt.ylabel('number of customer')
plt.title('customer churn prediction visualization')

plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['churn=yes', 'churn=no'])
plt.legend()

mc_churn_no = df1[df1.Churn == 'No'].MonthlyCharges
mc_churn_yes = df1[df1.Churn == 'Yes'].MonthlyCharges

plt.xlabel('Monthly Charges')  # # visualization of churn based on Monthly Charges
plt.ylabel('number of customer')
plt.title('customer churn prediction visualization')

plt.hist([mc_churn_yes, mc_churn_no], color=['green', 'red'], label=['churn=yes', 'churn=no'])
plt.legend()


def print_unique_col_values(df):  # we can see there is 'no' and 'no internet service' which is the same ,
    # so we replace it with no
    for column in df:
        if df[column].dtype == 'object':
            print(column, df[column].unique())


df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)  # replace these with NO because it is the same

# now we have to convert strings into numbers or numeric categories:
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

# now we one hot encoding for columns that have more than 2 categories:
df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
# now all data types are numbers

# in deep learning scaling step is very important:

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])  # now our data is scaled between 0 and 1

# now data cleaning is over and our dataframe is ready for modeling:

from sklearn.model_selection import train_test_split as tts

x = df2.drop('Churn', axis='columns')
y = df2['Churn']

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=5)

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test, y_test)  # model has 77% accuracy which is okay for now

yp = model.predict(x_test)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# now we want to do classification report:
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))

# now we want confusion matrix:

import seaborn as sn

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# 1. under sampling majority class: we have n number of fraud, we choose n number of non-fraud samples randomly then mix
# them with fraud samples and combine them together then train the model # not the best approach because we are
# trowing away so many data

# 2. over sampling minority class by duplication: we duplicate our fraud data until the number is equal to non-fraud
# data then train the model

# 3. over sampling minority class using SMOTE: 1- generate synthetic examples using k nearest neighbors algo ,
# 2- SMOTE: synthetic minority over-sampling technique

# 4. ensemble method: we n number of fraud data , then we choose n number of non-fraud sample and mix with fraud and
# build model1, then continue with another n number sample doing the same thing, then we have (non-fraud / n) models,
# then we use majority vote

# 5. focal loss: focal loss will penalize majority samples during loss calculation and give more weight to
# minority class samples


def ANN(x_train, y_train, x_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(26, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    if weights == -1:
        model.fit(x_train, y_train, epochs=100)
    else:
        model.fit(x_train, y_train, epochs=100, class_weight=weights)

    print(model.evaluate(x_test, y_test))

    y_preds = model.predict(x_test)
    y_preds = np.round(y_preds)

    print('classification report: \n', classification_report(y_test, y_preds))

    return y_preds

# goal here is to improve f1-score in classes
# we can see in our data : for first class there is 1033 samples and for second class there is 374 samples
# so there is imbalance in our data
# we are taking 347 samples for our model to balance the data

count_class_0, count_class_1 = df2.Churn.value_counts()

df_class_0 = df2[df2['Churn'] == 0]
df_class_1 = df2[df2['Churn'] == 1]

# now we do under sample:
df_class_0_under = df_class_0.sample(count_class_1)  # it will select n number of random samples
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)  # it will combine to dataframes
# now both classes having same number of samples
x = df_test_under.drop('Churn', axis='columns')
y = df_test_under['Churn']

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=15, stratify=y)  # stratify is for making
# sure we have balanced samples # it will split data randomly that test and train are having balanced classes

y_preds = ANN(x_train, y_train, x_test, y_test, 'binary_crossentropy', -1) # we can see precision and recall and
# f1-score is improved for second class, f1-score for first class is decreased

# now we do oversampling:

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_1_over, df_class_0], axis=0)
x = df_test_over.drop('Churn', axis='columns')
y = df_test_over['Churn']

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=15, stratify=y)
y_preds_over = ANN(x_train, y_train, x_test, y_test, 'binary_crossentropy', -1)

# now we are doing SMOTE:

x = df2.drop('Churn', axis='columns')
y = df2['Churn']

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')

x_sm, y_sm = smote.fit_resample(x,y)

x_train, x_test, y_train, y_test = tts(x_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

y_preds_smote = ANN(x_train, y_train, x_test, y_test, 'binary_crossentropy', -1)
# there so much improve in accuracy and f1-score


# now we use ensemble with under sampling:
# we split data into n batches : (n = majority data / minority data

x = df2.drop('Churn', axis='columns')
y = df2['Churn']

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=15, stratify=y)

df3 = x_train.copy()
df3['Churn'] = y_train

df3_class0 = df3[df3.Churn ==0]
df3_class1 = df3[df3.Churn ==1]


def get_train_batch(df_majority, df_minority, start, end):
    df_train = pd.concat([df_majority[start:end], df_minority], axis=0)

    x_train = df_train.drop('Churn', axis='columns')
    y_train = df_train.Churn

    return x_train, y_train


X_train, y_train = get_train_batch(df3_class0, df3_class1, 0, 1495)
y_pred1 = ANN(X_train, y_train, x_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 1495, 2990)
y_pred2 = ANN(X_train, y_train, x_test, y_test, 'binary_crossentropy', -1)

X_train, y_train = get_train_batch(df3_class0, df3_class1, 2990, 4130)
y_pred3 = ANN(X_train, y_train, x_test, y_test, 'binary_crossentropy', -1)

y_pred_final = y_pred1.copy()
for i in range(len(y_pred1)):
    n_ones = y_pred1[i] + y_pred2[i] + y_pred3[i]
    if n_ones>1:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0

cl_rep = classification_report(y_test, y_pred_final)
print(cl_rep)