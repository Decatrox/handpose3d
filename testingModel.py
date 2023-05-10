from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
#from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Masking
import numpy as np
import tensorflow as tf

actions = np.array(['rotatec', 'zoomin', 'zoomout'])
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(20, 63)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 63)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model.built = True
# model.build(input_shape=(None, 12, 63))

#print(model.summary())
# print(model.get_config())
#model2 = load_model('action.h5')
model.load_weights('actionHighAcc.h5')
#print(model.summary())
a = np.loadtxt('TestingFile-Nothing')
res = model.predict(np.expand_dims(a, axis=0))[0]
print(actions[np.argmax(res)])

rrr = [7, 0, 1, 2, 3, 4, 5]
print(rrr[-4:])
rrr.pop(0)
rrr.remove(0)
print(rrr)