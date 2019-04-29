# <div style="text-align:center">
# 
# # Final Project (DATS 6202 - O10), Spring 2019
# 
# ### Earthquake Time Prediction
# 
# ### Data Science, Columbian College of Arts & Sciences, George Washington University
# 
# ### Author: Elie Tetteh-Wayoe, Mihir Gadgil and Poornima Joshi
# </div>

# ## Introduction

# #### Problem and Motivation:
# 
# Forecasting earthquakes is one of the most important challenges in Earth science because
# of their devastating consequences. Current scientific studies related to earthquake
# forecasting focus on three key points: when the event will occur, where it will occur, and how
# large it will be. Los Alamos National Laboratory is hosting a [Kaggle competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction) to further
# this research.
# 
# In this competition, the aim is to address when the earthquake will take place. Specifically,
# predict the time remaining before laboratory earthquakes occur from seismic data (the data is generated by an experiment, it isn't actual seismic data).
# The challenge is that the data has only one feature and target to work with. The
# `acoustic_data` is the feature and `time_to_failure` is the target. Creating multiple sensible
# features from the available data will be a core part of the project.
# 
# If this challenge is solved and the physics are ultimately shown to scale from the laboratory
# to the field, researchers will have the potential to improve earthquake hazard assessments
# that could save lives and billions of dollars in infrastructure.

# ## Experiment
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isfile
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Set Pandas precision
pd.set_option('display.precision', 9)
# matplotlib inline plotting
# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

# ## EDA

# What kind of data do we have
print(os.listdir("data/"))

# How does the data look like 

z = pd.read_csv("data/train.csv", nrows=5)
print("The data looks like this :")
print(z.head())

# The code below has been commented out to avoid reading 9 GB data multiple time, for the sole purpose of counting its length. But it can be run to verify the number we have provided.
# 
# Total number of rows: 629,145,480

# Look at how big our data is

# df_length = 0
# for training in pd.read_csv('data/train.csv', chunksize=150000):
#     df_length = df_length + len(training)
    
# print("Train has: rows: {} ".format(df_length))


# We have one long array of seismic data. We will break it down into chunks of size 150,000 (chunk) and each chunk will be one signal in our data. The reasoning is that each segment in the test data has length 150,000. The `time_to_failure` at the last time step of each segment becomes the target associated with that segment.

# The code below is meant for plotting the data. It is again commented to avoid reading huge amounts of data. It should be uncommented if desired.
# An image of the plot has been included with the report

# %%time
# df_train = pd.DataFrame(columns=['acoustic_data', 'time_to_failure'], dtype=np.float)

# for train in pd.read_csv('data/train.csv', chunksize=150000):
#     df_train = df_train.append(train[::50])

# fig, ax1 = plt.subplots(figsize=(16, 8))
# plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
# plt.plot(df_train['acoustic_data'], color='b')
# ax1.set_ylabel('acoustic_data', color='b')
# plt.legend(['acoustic_data'])
# ax2 = ax1.twinx()
# plt.plot(df_train['time_to_failure'], color='g')
# ax2.set_ylabel('time_to_failure', color='g')
# plt.legend(['time_to_failure'], loc=(0.875, 0.9))
# plt.grid(False)


# ## Feature Engineering
# 
# Since the data we use here has only one feature to use for learning and the data set is fairly huge to work with, it is important to capture the essence of the data. Thus, we are generating more features using the exsisting data by using methods like calulating mean, standard deviation, rolling statistics etc.
# 
# Further, to choose the best features that contribute significantly to model, we build a random forest regressor in order to identify the top contributing features.

# ### Extracting features from each part of the segment
# 
# The original long seismic signal has been broken down into several more features. Usually features such as mean, standard deviation, range, percentiles etc are calculated over each part of the chunk and each part of the chunk is represented by its own list of such features.

# Creating more features from the existing data
n = 150_000
freq = 500
columns = [
    'mean', 'std', 'min', 'max', 'sum', 'abs_mean', 'abs_std', 'abs_max', 'abs_sum', 'argmax', 'rate_mean', 'rate_std',
    'rate_max', 'rate_min', 'rate_abs_max'
]

columns.extend(['fftr' + str(i) for i in range(0, freq)])
columns.extend(['fftr' + str(i) for i in range(n//2 - freq, n//2 + freq)])
columns.extend(['fftr' + str(i) for i in range(n-freq, n)])
columns.extend(['ffti' + str(i) for i in range(0, freq)])
columns.extend(['ffti' + str(i) for i in range(n//2 - freq, n//2 + freq)])
columns.extend(['ffti' + str(i) for i in range(n-freq, n)])

roll_windows = [100, 500, 1000, 2000, 4000, 10000]
columns.extend(['rolling_mean_' + str(i) for i in roll_windows])
columns.extend(['rolling_std_' + str(i) for i in roll_windows])

df_train = pd.DataFrame(dtype=np.float, columns=columns)


def generate_features(chunk):
    mean = chunk['acoustic_data'].mean()
    std = chunk['acoustic_data'].std()
    min = chunk['acoustic_data'].min()
    max = chunk['acoustic_data'].max()
    sum = chunk['acoustic_data'].sum()
    abs_sum = chunk['acoustic_data'].abs().sum()
    abs_max = chunk['acoustic_data'].abs().max()
    abs_mean = chunk['acoustic_data'].abs().mean()
    abs_std = chunk['acoustic_data'].abs().std()
    argmax = chunk['acoustic_data'].abs().values.argmax()
    rate = np.diff(chunk['acoustic_data'].values)
    rate_mean = rate.mean()
    rate_std = rate.std()
    rate_max = rate.max()
    rate_min = rate.min()
    rate_abs_max = np.abs(rate).max()
    fft = np.fft.fft(chunk['acoustic_data'], n=n)
    result = [
        mean, std, min, max, sum, abs_mean, abs_std, abs_max, abs_sum, argmax, rate_mean, rate_std, rate_max, rate_min,
        rate_abs_max
    ]
    result.extend(list(fft.real[0:freq]))
    result.extend(list(fft.real[n//2-freq:n//2+freq]))
    result.extend(list(fft.real[n-freq:n]))
    result.extend(list(fft.imag[0:freq]))
    result.extend(list(fft.imag[n//2-freq:n//2+freq]))
    result.extend(list(fft.imag[n-freq:n]))
    for window in roll_windows:
        result.append(
            chunk['acoustic_data'].rolling(window=window).mean().mean(skipna=True)
        )
        result.append(
            chunk['acoustic_data'].rolling(window=window).std().mean(skipna=True)
        )
    return result


i = 0
for chunk in pd.read_csv('data/train.csv', chunksize=n):
    df_train.loc[i, columns] = generate_features(chunk)
    df_train.loc[i, 'time_to_failure'] = chunk['time_to_failure'].values[-1]
    i += 1

print(df_train.describe())

print(df_train.head())


# Check if any nan values are generated
df_train.isna().sum().sum()

# Seperate the data into X_train and Y_train

X_train = df_train.drop(columns=['time_to_failure']).values
y_train = df_train['time_to_failure'].values

# Choosing the best contributing features using random forest regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=0, max_features='sqrt', n_jobs=-1)
pipe_rfr = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestRegressor', rfr)])

# Calling fit on the random forest regressor
pipe_rfr.fit(X_train, y_train)

# Creating a dataframe to choose the best features
features = pd.DataFrame({'Feature': columns, 'Importance': rfr.feature_importances_, 'Correlation': df_train.drop(columns='time_to_failure').corrwith(df_train['time_to_failure']).abs().values})

# Sorting the features in descending order so we choose the best one
features = features.sort_values(by='Importance', ascending=False)
print(features.head())

# Visualizing the best performing feature from the 'features' dataframe
# plt.figure(figsize=(16, 9))
# plt.barh(y='Feature', width='Importance', data=features[:30])
# plt.show()

# Subsetting the top 27 features
n_features = 30
X_train = df_train[features['Feature'][:n_features]].values
y_train = df_train['time_to_failure'].values


# #### Linear Regression 
# We started with simple linear regression, since that is the simplest and most straight forward method we are familiar with

# Prepare pipeline

pipe_linear = Pipeline([('StandardScaler', StandardScaler()), ('Linear', linear_model.LinearRegression())])

# Hyperparamters for the linear model

parameters_linear = [{
    'Linear__fit_intercept': ('True', 'False'),
    'Linear__normalize': ('True', 'False'),
    'Linear__copy_X': ('True', 'False')
}]

# Perform grid search CV on with different parameters

gs_linear = GridSearchCV(
    estimator=pipe_linear,
    param_grid=parameters_linear,
    iid=False,
    n_jobs=-1,
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )
)

gs_linear.fit(X_train, y_train)
print('Best score:', gs_linear.best_score_)
print('Best hyperparameters:', gs_linear.best_params_)


# #### Elastic Net
# To improve the Linear regression results, we used a penalised method like Elastic search. As we can notice, we got slightly better results

# Prepare pipeline

pipe_elastic = Pipeline([('StandardScaler', StandardScaler()), ('ElasticNet', ElasticNet(random_state=0))])

# Hyperparamters for the elasticnet model

param_grid_elastic = [{
    'ElasticNet__max_iter': [1, 5, 10],
    'ElasticNet__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'ElasticNet__l1_ratio': np.arange(0.0, 1.0, 0.1)
}]

# Perform grid search CV on with different parameters
gs_elastic = GridSearchCV(
    estimator=pipe_elastic,
    param_grid=param_grid_elastic,
    iid=False,
    n_jobs=-1,
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )
)

gs_elastic.fit(X_train, y_train)
print('Best score:', gs_elastic.best_score_)
print('Best hyperparameters:', gs_elastic.best_params_)


# #### Neural Network

# The next model we are trying is neural network. Earthquakes are a complicated phenomenon, so we expect a neural network to be better at capturing the non-linearity and perform better than linear regression.
# 
# We are using the top 27 features ranked by importance, so we search through a grid of different hidden layer numbers and sizes. The other hyperparameters being tuned are the learning rate, regularization parameter and the tolerance for cross validation score stopping.

# Prepare Pipeline

pipe_nn = Pipeline([('StandardScaler', StandardScaler()), ('Regressor', MLPRegressor(random_state=0))])

# Hyperparamters for the neural network model

param_grid_nn = [{
    'Regressor__hidden_layer_sizes': [
        (int(n_features*2/3)),
        (int(n_features*2/3), int(int(n_features*2/3)*2/3)),
        (int(n_features*2/3), int(int(n_features*2/3)*2/3), int(int(int(n_features*2/3)*2/3)*2/3))],
    'Regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    'Regressor__learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
    'Regressor__tol': [0.0001, 0.001, 0.01]
}]

# Perform grid search CV on with different parameters

gs_nn = GridSearchCV(
    estimator=pipe_nn,
    param_grid=param_grid_nn,
    iid=False,
    n_jobs=-1,
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )
)

gs_nn.fit(X_train, y_train)
print('Best score:', gs_nn.best_score_)
print('Best hyperparameters:', gs_nn.best_params_)


# #### Neural Network using tensorflow

# Create function returning a compiled network
def kerasModel(optimizer, metrics, loss, activation, input_shape= X_train.shape[1]):
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=input_shape, activation=activation, input_dim=input_shape))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units= int(input_shape/2), activation=activation))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1))

    # Compile neural network
    network.compile(loss=loss, # Cross-entropy
                    optimizer=optimizer, # Optimizer
                    metrics=[metrics]) # Accuracy performance metric
    
    # Return compiled network
    return network


KModel = KerasRegressor(build_fn=kerasModel, verbose=0)

# Create hyperparameter space
epochs = [50 , 100]
batches = [50, 100]
optimizer = ['adam','sgd']
loss = ['mse']
activation = ['relu', 'exponential']
metrics = ['mse']

# Create hyperparameter options
hyperparameters = dict(Model__optimizer=optimizer, Model__loss = loss, Model__epochs=epochs, Model__batch_size=batches, Model__activation= activation,
                            Model__metrics = metrics)


pipe_keras = Pipeline([('StandardScaler', StandardScaler()), ('Model', KModel)])

gs_keras = GridSearchCV(
    estimator=pipe_keras,
    param_grid=hyperparameters,
    iid = False,
    n_jobs=-1,
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )
)

gs_keras.fit(X_train, y_train)
print('Best score:', gs_keras.best_score_)
print('Best hyperparameters:', gs_keras.best_params_)

# Traverse through the test directory
path = 'data/test/'
files = [f[:-4] for f in listdir(path) if isfile(path + f) and f[-3:] == 'csv']

# Prepare a submission dataframe

predictions = pd.DataFrame(index=files, dtype=np.float, columns=['time_to_failure'])
predictions.index.name = 'seg_id'
predictions_Keras = pd.DataFrame(index=files, dtype=np.float, columns=['time_to_failure'])
predictions_Keras.index.name = 'seg_id'

# For all files in the test folder, run predict function and add to 'predictions' dataframe

for f in files:
    df = pd.read_csv(path+f+'.csv')
    df_test = pd.DataFrame(np.array(generate_features(df)).reshape(1,-1), columns=columns)
    X_test = df_test[features['Feature'][:n_features]].values
    y = gs_keras.predict(X_test)
    predictions_Keras.loc[f, 'time_to_failure'] = y
    y = gs_nn.predict(X_test)[0]
    predictions.loc[f, 'time_to_failure'] = y

# Peek at the predictions

predictions_Keras.head()

predictions.head()

# Export the predictions dataframe to a csv

predictions_Keras.to_csv('submission_Keras.csv')
predictions.to_csv('submission.csv')


# | Model | Best Parameter | Best Score |
# |:--- |:--- | ---:|
# | Linear Regression | 'Linear__copy_X': 'True', 'Linear__fit_intercept': 'True', 'Linear__normalize': 'True' | 0.277 |
# | Elastic Net | 'ElasticNet__alpha': 0.01, 'ElasticNet__l1_ratio': 0.9, 'ElasticNet__max_iter': 10 | 0.328 |
# | Neural Network | 'Regressor__alpha': 0.1, 'Regressor__hidden_layer_sizes': 20, 'Regressor__learning_rate_init': 0.01, 'Regressor__tol': 0.0001 | 0.444 |
# | Neural Network (Using Keras)| 'Model__activation': 'relu', 'Model__batch_size': 50, 'Model__epochs': 100, 'Model__loss': 'mse', 'Model__metrics': 'mse', 'Model__optimizer': 'adam' | 7.606 (MSE) |

# ### Conclusion

# Based on the cross validation score, we expect Neural Network with 1 input layer, 1 hidden layer with 18 nodes and 1 output layer to give us the best R<sup>2</sup> coefficient. 
# 
# Another important observation we can make in this experiment is that, the performance of the model is higly correlated with the features used. As we can see in the graph, rolling_mean_500 has an importance value of 0.41, followed by  rolling_mean_2000 with an importance value of 0.01. We can see a strike difference in the performance of the features.
# 
# In conclusion, Neural Network is the best model for this data. More importantly, efficient feature engineering is the key to building a good model.

# ### References
# 
# - https://kaggle.com/c/LANL-Earthquake-Prediction/discussion
# - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
# - https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

