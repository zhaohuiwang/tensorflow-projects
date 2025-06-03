


import tensorflow as tf

seq_length = 3

# univariate
x = tf.range(25, dtype=tf.float32)[:-seq_length]
y = tf.range(25, dtype=tf.float32)[seq_length:]

'''
# multivariate
x = tf.concat([
    # example feature 1
    tf.reshape(tf.range(25, dtype=tf.float32)[:-seq_length], (-1, 1)),
    # example feature 2
    tf.reshape(tf.linspace(0., .24, 25)      [:-seq_length], (-1, 1))], axis=-1)

y = tf.concat([
    # example lable 1
    tf.reshape(tf.range(25, dtype=tf.float32)[seq_length:], (-1, 1)),
    # example lable 2
    tf.reshape(tf.linspace(0., .24, 25)      [seq_length:], (-1, 1))], axis=-1)
'''
# Creates a dataset of sliding windows over a timeseries provided as array.
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
   data=x, targets=y,
   sequence_length=seq_length,
   batch_size=1)

# for present_values, next_value in ds.take(5):
for present_values, next_value in ds:
    print(tf.squeeze(present_values).numpy(), '-->', next_value.numpy())

print(
    f'Input shape: \t{present_values.shape}\n'
    f'Output shape: \t{next_value.shape}'
)
# ============= Prep ===========

import tensorflow as tf
import numpy as np

batch_size = 32
window_size = 5

#creating numpy data structures representing the problem
X = np.random.random((100,5))
y = np.random.random((100))
src = np.expand_dims(np.array([0]*50 + [1]*50),1)

#appending source information to X, for filtration
X = np.append(src, X, 1)

#making a time series dataset which does not respect src
Xy_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X, targets=y,
    sequence_length=window_size, batch_size=1, 
    sequence_stride=1, shuffle=True)

#filtering by and removing src info
def single_source(x,y):
    source = x[:,:,0]
    return tf.reduce_all(source == source[0])
    
def drop_source(x,y):
    x_ = x[:, :, 1:]
    print(x_)
    return x_, y

Xy_ds = Xy_ds.filter(single_source)
Xy_ds = Xy_ds.map(drop_source)
Xy_ds = Xy_ds.unbatch().batch(batch_size)

#printing the dataset
i = 0
for x, y in enumerate(Xy_ds):
    i+=1
    print(x)
    print(y)
print(f'total batches: {i}')


# ============= Train ===========
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, LSTM, Flatten

#training a model, to validate the dataset is working correctly
model = Sequential()
model.add(InputLayer(input_shape=[window_size,5]))
model.add(LSTM(units=3))
model.add(Flatten())
model.add(Dense(units=1, activation='relu'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer, metrics=['accuracy'])

history = model.fit(Xy_ds,epochs=1)

