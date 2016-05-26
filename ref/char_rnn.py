"""
Character-level text generation.

This code divide a long character string to chunks of 200 characters,
and it learns a model for the next character given the previous ones.
At the end it inefficiently generates 128 sentences, each of 200 chars.
https://github.com/fchollet/keras/issues/197
"""

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM
# sys.path.append('/home/USER/python/keras/')

# Obtain the corpus of character sequence to train from.
# Here it is just the sequence 123456789 repeated 100000 times.
x = "123456789"*100000

# Construct a dictionary, and the reverse dictionary
# for the participating chars.
# '*" is a 'start-sequence' character.
dct = ['*'] + list(set(x))
max_features = len(dct)
rev_dct = [(j, i) for i, j in enumerate(dct)]
rev_dct = dict(rev_dct)

# Convert the characters to their dct indexes.
x = [rev_dct[ch] for ch in x]

# Divide the corpuse to substrings of length 200.
n_timestamps = 200
x = x[:len(x)-len(x) % n_timestamps]
x = np.array(x, dtype='int32').reshape((-1, n_timestamps))

# Generate input and ouput per substring, as an indicator matrix.
y = np.zeros((x.shape[0], x.shape[1], max_features), dtype='int32')
for i in np.arange(x.shape[0]):
    for j in np.arange(x.shape[1]):
        y[i, j, x[i, j]] = 1

# Shift-1 the input sequences to the right, and make them start with '*'.
x = np.roll(y, 1, axis=1)
x[:, 0, :] = 0
x[:, 0, 0] = 1

# print x.shape, y.shape
# print x[:, 1, :]
# print y[:, 0, :]
# sys.exit(1)

# Build the model.
model = Sequential()
model.add(LSTM(output_dim=256, input_dim=max_features,
          return_sequences=True))
# model.add(LSTM(256, return_sequences=True))
# model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(max_features, activation='softmax')))
# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(x, y, batch_size=64, nb_epoch=10)


# Sample 128 sentences (200 characters each) from model.
def mnrnd(probs):
    rnd = np.random.random()
    for i in xrange(len(probs)):
        rnd -= probs[i]
        if rnd <= 0:
            return i
    return i

sentences = np.zeros((128, n_timestamps+1, max_features))
sentences[:, 0, 0] = 1

# Start sampling char-sequences. At each iteration i the probability over
# the i-th character of each sequences is computed.
for i in np.arange(n_timestamps):
    probs = model.predict_proba(sentences)[:, i, :]
    # Go over each sequence and sample the i-th character.
    for j in np.arange(len(sentences)):
        sentences[j, i+1, mnrnd(probs[j, :])] = 1
sentences = [sentence[1:].nonzero()[1] for sentence in sentences]

# Convert to readable text.
text = []
for sentence in sentences:
    text.append(''.join([dct[word] for word in sentence]))

print text
