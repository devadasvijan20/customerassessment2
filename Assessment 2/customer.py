# -*- coding: utf-8 -*-
"""


@author: User
"""

import pandas
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
 
PATH = os.path.join(os.getcwd(),'new_customers.csv')
df = pd.read_csv(PATH)

df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

print(df.isnull().sum())
# filling with most common class
df_clean = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
print(df_clean.isnull().sum())
# now no data has null values.
df_clean['Ever_Married']=df_clean['Ever_Married'].map({'No':0, 'Yes':1})
df_clean['Graduated']=df_clean['Graduated'].map({'No':0, 'Yes':1})
# dropping columns not necessary  for the analysis
df_clean=df_clean.drop(['ID','Var_1'],axis=1)
# encode class values as integers
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
df_clean['Profession']= label_encoder.fit_transform(df_clean['Profession'])
df_clean['Spending_Score']= label_encoder.fit_transform(df_clean['Spending_Score'])
df_clean['Segmentation']= label_encoder.fit_transform(df_clean['Segmentation'])
print(df_clean.head())

X=df_clean[['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession','Work_Experience', 'Spending_Score', 'Family_Size']]
Y=df_clean['Segmentation']
encoder = LabelEncoder()
encoder.fit(Y)
y = encoder.transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
history = model.fit(X_train_scaled, y_train, epochs=100)
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(
    np.arange(1, 101), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 101), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();



# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)