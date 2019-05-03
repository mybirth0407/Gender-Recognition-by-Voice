# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Feature scaling
from sklearn.model_selection import KFold # Use k-fold

from keras import metrics # Keras metrics method
from keras.layers import Dense # Keras perceptron neuron layer implementation.
from keras.layers import Dropout # Keras Dropout layer implementation.
from keras.layers import Activation # Keras Activation Function layer implementation.
from keras.models import Sequential # Keras Model object.
from keras import optimizers # Keras Optimizer for custom user.
from keras import losses # Keras Loss for custom user.

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


def plot(model):
    from keras.utils.vis_utils import plot_model
    import os
    plot_model(model, 'plot.png', show_shapes=True)

def create_model(fl, lr=0.01):
    """ Create Neural Networks Model
    @fl: feature length(num of input features)
    @lr: learning rate(step size), default value=0.01 
    """
    model = Sequential()
    model.add(Dense(fl, input_dim=fl, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(22, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(22, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(22, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    plot(model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
def train_and_evaluate(model, epochs, batch_size, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # evaluate the model
    score = model.evaluate(x_test, y_test)
    predict = model.predict(x_test, batch_size=batch_size)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    return predict, score


# print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('./voice.csv')
scaler = StandardScaler().fit(df[df.columns[0: -1]])
scaling_features = scaler.transform(df[df.columns[0: -1]])

d = {'male': 1, 'female': 0}
labels = df[df.columns[-1]].map(d)

FV_LEN = len(scaling_features[0])
k = 2
epochs = 300
batch_size = 32
learning_rate = 0.0003

skf = KFold(n_splits=k, shuffle=True, random_state=None)

predicts = [0] * k
scores = [0] * k

for i, (train, test) in zip(range(k), skf.split(scaling_features)):
    model = None
    model = create_model(fl=FV_LEN ,lr=learning_rate)
    
    predicts[i], scores[i] = train_and_evaluate(
        model, epochs, batch_size, scaling_features[train],
        labels[train], scaling_features[test], labels[test])

# Any results you write to the current directory are saved as output.
print(scores)
