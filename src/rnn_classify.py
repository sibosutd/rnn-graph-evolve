"""
Classifying graphs using RNN.

Model: BA and RANDOM
"""

import numpy as np
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import glob

glob.set_dir()

# read data
sequences_er = np.load(glob.RESULT_PATH+'BA.npy')
sequences_ba = np.load(glob.RESULT_PATH+'RANDOM.npy')
X = np.concatenate((sequences_er, sequences_ba), axis=0)

NUM_OF_NODE = X.shape[1]

# cross validation
y = np.kron(np.eye(2), np.ones((1000, 1)))
X_train, X_test, y_train, y_test = cross_validation\
    .train_test_split(X, y, test_size=0.1)
print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

# LSTM Classification
model = Sequential()
# model.add(Embedding(max_features, 256, input_length=maxlen))
model.add(LSTM(output_dim=32, input_dim=100, return_sequences=True,
          activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(LSTM(32, activation='sigmoid',
          inner_activation='hard_sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=32, nb_epoch=50,
          verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# Y_pred = model.predict_classes(X_test, verbose=1)

# print(Y_pred)

# if n == 0:
#     cm = confusion_matrix(Y_test_vec, Y_pred, np.arange(20))
#     accuracy = score[1]
# else:
#     cm += confusion_matrix(Y_test_vec, Y_pred, np.arange(20))
#     accuracy += score[1]

# cm_svc_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# ######################################################
# # print 'Sensor:', SENSOR
# print 'accuracy:', accuracy / 10.0

# # Write file to csv file with precision control
# np.savetxt(RESULT_PATH+'cm_smooth_3_early_lstm_1_layer.csv',
#            cm_svc_normalized, delimiter=',')
