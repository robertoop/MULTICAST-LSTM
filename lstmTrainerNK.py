import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

text = open("HandGunTrayectory_TOTAL.txt", 'r')
count = len(text.readlines())
print(count)
text = open("HandGunTrayectory_TOTAL.txt", 'r')
dataset = np.zeros((count, 4))
timestep = np.zeros((count, 1))
i = 0
for x in text:
    coords = (x.split(',')[0:6])
    print(coords)
    numero = int(coords[0])
    ymin = float(coords[1])
    xmin = float(coords[2])
    ymax = float(coords[3])
    xmax = float(coords[4])
    dataset[i, :] = [ymin, xmin, ymax, xmax]
    timestep[i] = [numero]
    i = i + 1
print(dataset.shape)
print(timestep.shape)


def sp_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps = 5
n_features = 4
X, y = sp_sequences(dataset, n_steps)
print(X.shape, y.shape)

model = Sequential()
model.add(LSTM(200, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(200, activation='tanh'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000)

x_input = array(
    [[0.32, 0.31, 0.41, 0.41], [0.44, 0.42, 0.54, 0.55], [0.51, 0.54, 0.64, 0.68], [0.61, 0.62, 0.73, 0.72], [0.71, 0.73, 0.87, 0.89]])
x_input = x_input.reshape((1, n_steps, n_features))

model.save("Move_lstm3RM.h5")
print(x_input)
yhat = model.predict(x_input, verbose=0)
print(yhat)


