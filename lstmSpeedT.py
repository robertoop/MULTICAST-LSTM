import time
import numpy as np

from tensorflow.keras.models import load_model

lstmM = load_model("Move_lstm3RM.h5")

x_input = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
x_input = x_input.reshape((1, 5, 4))


for i in range(0,100):
    st_time = time.time ()
    next = lstmM.predict (x_input, verbose=0)

    e_time = time.time ()
    el_time = e_time - st_time
    print ('Complete in : ' + str (el_time) + 's')