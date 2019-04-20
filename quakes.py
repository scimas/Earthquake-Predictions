import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile

# Pandas precision
pd.set_option('display.precision', 9)
# 629_145_480 number of rows

n = 150_000
freq = 500
columns = [
    'mean', 'std', 'min', 'max', 'sum', 'abs_mean', 'abs_std', 'abs_max', 'abs_sum', 'argmax', 'rate_mean', 'rate_std',
    'rate_max', 'rate_min', 'rate_abs_max'
]
# columns.extend(['fftr' + str(i) for i in range(0, 1000)])
# columns.extend(['fftr' + str(i) for i in range(n//2 - 1000, n//2)])
# columns.extend(['ffti' + str(i) for i in range(0, 1000)])
# columns.extend(['ffti' + str(i) for i in range(n//2 - 1000, n//2)])
# columns.extend(['fftr' + str(i) for i in range(n//2, n//2 + 1000)])
# columns.extend(['fftr' + str(i) for i in range(n - 1000, n)])
# columns.extend(['ffti' + str(i) for i in range(n//2, n//2 + 1000)])
# columns.extend(['ffti' + str(i) for i in range(n - 1000, n)])
columns.extend(['fftr' + str(i) for i in range(0, freq)])
columns.extend(['fftr' + str(i) for i in range(n//2 - freq, n//2 + freq)])
columns.extend(['fftr' + str(i) for i in range(n-freq, n)])
columns.extend(['ffti' + str(i) for i in range(0, freq)])
columns.extend(['ffti' + str(i) for i in range(n//2 - freq, n//2 + freq)])
columns.extend(['ffti' + str(i) for i in range(n-freq, n)])

roll_windows = [100, 500, 1000, 2000, 4000, 10000]
columns.extend(['rolling_mean_' + str(i) for i in roll_windows])
columns.extend(['rolling_std_' + str(i) for i in roll_windows])

df = pd.read_csv('data/train.csv')

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
#     result.extend(list(fft.real[0:1000]))
#     result.extend(list(fft.real[n//2-1000:n//2]))
#     result.extend(list(fft.imag[0:1000]))
#     result.extend(list(fft.imag[n//2-1000:n//2]))
#     result.extend(list(fft.real[n//2:n//2+1000]))
#     result.extend(list(fft.real[n-1000:n]))
#     result.extend(list(fft.imag[n//2:n//2+1000]))
#     result.extend(list(fft.imag[n-1000:n]))
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

# for i in range(len(df)//n):
#     df_train.loc[i, columns] = generate_features(df[i*n:(i+1)*n])
#     df_train.loc[i, 'time_to_failure'] = df['time_to_failure'].values[-1]

i = 0
for chunk in pd.read_csv('data/train.csv', chunksize=n):
    df_train.loc[i, columns] = generate_features(chunk)
    df_train.loc[i, 'time_to_failure'] = chunk['time_to_failure'].values[-1]
    i += 1

# df_train.head()

X_train = df_train.drop(columns=['time_to_failure']).values
y_train = df_train['time_to_failure'].values

rfr = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
pipe_rfr = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestRegressor', rfr)])

pipe_rfr.fit(X_train, y_train)

features = pd.DataFrame({'Feature': columns, 'Importance': rfr.feature_importances_, 'Correlation': df_train.drop(columns='time_to_failure').corrwith(df_train['time_to_failure']).abs().values})

# features = features.sort_values(by='Correlation', ascending=False)
features = features.sort_values(by='Importance', ascending=False)
# features

# plt.figure(figsize=(16, 9))
# plt.barh(y='Feature', width='Importance', data=features[:100])
# plt.show()
n_features = 51
X_train = df_train[features['Feature'][:n_features]].values
y_train = df_train['time_to_failure'].values

pipe = Pipeline([('StandardScaler', StandardScaler()), ('Regressor', MLPRegressor(random_state=0, max_iter=300))])

param_grid = [{
    'Regressor__hidden_layer_sizes': [(34), (34, 22), (34, 22, 14)],
    'Regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    'Regressor__learning_rate_init': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'Regressor__tol': [0.0001, 0.001, 0.01, 0.1]
}]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    iid=False,
    n_jobs=-1,
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )
)

gs.fit(X_train, y_train)
print('Best score:', gs.best_score_)
print('Best hyperparameters:', gs.best_params_)

path = 'data/test/'
files = [f[:-4] for f in listdir(path) if isfile(path + f)]

predictions = pd.DataFrame(index=files, dtype=np.float, columns=['time_to_failure'])
predictions.index.name = 'seg_id'

for f in files:
    df = pd.read_csv(path+f+'.csv')
    df_test = pd.DataFrame(np.array(generate_features(df)).reshape(1,-1), columns=columns)
    X_test = df_test[features['Feature'][:n_features]].values
    y = gs.predict(X_test)[0]
    predictions.loc[f, 'time_to_failure'] = y

predictions.to_csv('submission.csv')

# predictions

