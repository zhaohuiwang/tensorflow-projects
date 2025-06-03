
# https://www.tensorflow.org/tutorials/structured_data/time_series

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from dataprep import WindowGenerator
from model_utils import compile_and_fit

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003. For efficiency, you will use only the data collected between 2009 and 2016. The goal is to model hourly predictions
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
# i = slice(5, -1, 6); df[i]
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

df.describe().transpose()

# Inspect and cleanup
# Make it easier for the model to interpret if you convert the wind direction and velocity columns to a wind vector
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))



# split the data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# Normalized the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

"""
This tutorial builds a variety of models (including Linear, DNN, CNN and RNN models), and uses them for both:
Single-output, and multi-output predictions/labels (by specifying the lable_columns, if None then all features)
Single-time-step and multi-time-step predictions by adjusting the input_width and label_width values.

"""

# test example
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'],
                     train_df=train_df, val_df=val_df, test_df=test_df)
# total_window_size = input_width + shift = 7
w2

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: \t(batch, time, features)\n'
      f'Window shape: \t{example_window.shape}\n'
      f'Inputs shape: \t{example_inputs.shape}\n'
      f'Labels shape: \t{example_labels.shape}')
# Typically, data in TensorFlow is packed into arrays where the outermost index is across examples (the "batch" dimension). The middle indices are the "time" or "space" (width, height) dimension(s). The innermost indices are the features.

# The Dataset.element_spec property tells you the structure, data types, and shapes of the dataset elements. For example
w2.train.element_spec

# Iterating over a Dataset yields concrete batches
for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

############### Single step models ###############

single_step_window = WindowGenerator(
  train_df=train_df, val_df=val_df, test_df=test_df,
  input_width=1, label_width=1, shift=1,
  label_columns=['T (degC)'])
single_step_window

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)


# create a wider WindowGenerator that generates windows 24 hours of consecutive inputs and labels at a time. 
wide_window = WindowGenerator(
  train_df=train_df, val_df=val_df, test_df=test_df
  input_width=24, label_width=24, shift=1,
  label_columns=['T (degC)'])

print(f'{wide_window}\n'
      f'Input shape: \t{wide_window.example[0].shape}\n'
      f'Output shape: \t{baseline(wide_window.example[0]).shape}')

# The simplest trainable model you can apply to this task is to insert linear transformation between the input and output. A tf.keras.layers.Dense layer with no activation set is a linear model. The layer only transforms the last axis of the data from (batch, time, inputs) to (batch, time, units); it is applied independently to every item across the batch and time axes.
simple_linearR_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print(f'{wide_window}\n'
      f'Input shape: \t{single_step_window.example[0].shape}\n'
      f'Output shape: \t{simple_linearR_model(single_step_window.example[0]).shape}')

history = compile_and_fit(simple_linearR_model, single_step_window)

val_performance['Linear'] = simple_linearR_model.evaluate(single_step_window.val, return_dict=True)
performance['Linear'] = simple_linearR_model.evaluate(single_step_window.test, verbose=0, return_dict=True)

""""
# One advantage to linear models is that they're relatively simple to interpret. You can pull out the layer's weights and visualize the weight assigned to each input
fig, axis = plt.subplots(1, 1)
plt.bar(x = range(len(train_df.columns)),
        height=simple_linearR_model.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()
# Note: Sometimes the model doesn't even place the most weight on the input T (degC). This is one of the risks of random initialization.
"""

# Another model similar to the linear model, except it stacks several a few Dense layers between the input and the output:
dense_linearR_model = tf.keras.Sequential([
   tf.keras.layers.Dense(units=64, activation='relu'),
   tf.keras.layers.Dense(units=64, activation='relu'),
   tf.keras.layers.Dense(units=1)
   ])

history2 = compile_and_fit(dense_linearR_model, single_step_window)

val_performance['Dense'] = dense_linearR_model.evaluate(single_step_window.val, return_dict=True)
performance['Dense'] = dense_linearR_model.evaluate(single_step_window.test, verbose=0, return_dict=True)

# Create a WindowGenerator that will produce batches of three-hour inputs and one-hour labels
CONV_WIDTH = 3
conv_window = WindowGenerator(
    train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'])

# train a dense model on a multiple-input-step window by adding a tf.keras.layers.Flatten as the first layer of the model
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape((1, -1)),
])
# The main down-side of this approach is that the resulting model can only be executed on input windows of exactly this shape. The convolutional models in the next section fix this problem. The difference between the conv_model and the multi_step_dense model is that the conv_model can be run on inputs of any length. The convolutional layer is applied to a sliding window of inputs

print(f'{wide_window}\n'
      f'Input shape: \t{conv_window.example[0].shape}\n'
      f'Output shape: \t{multi_step_dense(conv_window.example[0]).shape}')

history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val, return_dict=True)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)

# Convolution neural network Note the changes: The tf.keras.layers.Flatten and the first tf.keras.layers.Dense are replaced by a tf.keras.layers.Conv1D. The tf.keras.layers.Reshape is no longer necessary since the convolution keeps the time axis in its output.
# keras.layers.Conv1D() creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. 
# 1d Convolution https://www.youtube.com/watch?v=yd_j_zdLDWs
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
# the dimension of the output space (the number of filters in the convolution)
       filters=32,
# int or tuple/list of 1 integer, specifying the size of the convolution window.
       kernel_size=(CONV_WIDTH,),
       activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])


print(f'Conv model on `conv_window`\n'
      f'Input shape: \t{conv_window.example[0].shape}\n'
      f'Output shape: \t{conv_model(conv_window.example[0]).shape}')

history = compile_and_fit(conv_model, conv_window)
val_performance['Conv'] = conv_model.evaluate(conv_window.val, return_dict=True)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)

conv_model.summary()

print("Wide window\n"
      f'Input shape: \t{wide_window.example[0].shape}\n'
      f'Labels shape: \t{wide_window.example[1].shape}\n'
      f'Output shape: \t{conv_model(wide_window.example[0]).shape}')


# # The main down-side of the multi_step_dense approach is that the resulting model can only be executed on input windows of exactly this shape.
print("Wide window\n"
      f'Input shape: \t{wide_window.example[0].shape}\n'
      f'Labels shape: \t{wide_window.example[1].shape}\n')
try:
  print(f'Output shape: \t{multi_step_dense(wide_window.example[0]).shape}')
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')


# Recurrent neural network
# RNNs process a time series step-by-step, maintaining an internal state from time-step to time-step.


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(units=32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print("Wide window\n"
      f'Input shape: \t{wide_window.example[0].shape}\n'
      f'Labels shape: \t{wide_window.example[1].shape}\n'
      f'Output shape: \t{lstm_model(wide_window.example[0]).shape}')

"""
tf.keras.layers.LSTM(
    units,
    dropout=0.0,
    return_sequences=False,
    stateful=False,
    ...
)
units: Positive integer, dimensionality of the output space.

dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.

return_sequences: Boolean. Whether to return the last output in the output sequence (when False), or the full sequence (When True). Default: False.
So when we set return_sequences=True, if your input is of size (batch_size, time_steps, input_size) then the LSTM output will be (batch_size, time_steps, output_size). This is called a sequence to sequence model because an input sequence is converted into an output sequence. If we set return_sequences=False the model returns the output state of only the last LSTM cell.

stateful: Boolean (default: False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch, So state information get propagated between batches. If stateful=False then the states are reset after each batch.

A functional model example
timesteps = 40  # dimensionality of the input sequence
features = 3    # dimensionality of the input vector feature
OD = 2          # dimensionality of the LSTM outputs (Hidden & Cell states) 
input = tf.keras.layers.Input(shape=(timesteps, features))
output= tf.keras.layers.LSTM(units=OD)(input)
model_LSTM = tf.keras.models.Model(inputs=input, outputs=output)
model_LSTM.summary()

print("Shapes of Matrices and Vecors: e.g. forget gate's function f_t = sigmoid(Wx_t + Ux_t + b_t) \n"
      f"\tInput [batch_size, timesteps, feature]: {input.shape}\n" 
      f"\tInput feature/dimension (x in formulations): {input.shape[2]}\n"
      f"\tNumber of Hidden States/LSTM units (cells)/dimensionality of the output space (h in formulations): {OD}")

print("Parameters and their values: \n"
      f"W: {model_LSTM.layers[1].get_weights()[0]}\n"
      f"U: {model_LSTM.layers[1].get_weights()[1]}\n"
      f"b: {model_LSTM.layers[1].get_weights()[2]}\n"
      f"Total number of paramters: {4*(features*OD + OD*OD + OD)}")
# There are total of 4 similar functions, each has their unique "W", "U", and "b". The same "W", "U", and "b" are shared throughout the time-steps, so timesteps of the input sequence does not affect the number of parameters.


"""


history = compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, return_dict=True)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

cm = lstm_model.metrics[1]
cm.metrics

for name, metrics in val_performance.items():
   print(f"{name:16s}", end='')
   for metric, value in metrics.items():
      print(f"\t{metric}:{value:0.4f}", end='')
   print()

metric_name = 'mean_absolute_error' # or 'loss'
for name, value in performance.items():
  print(f'{name:16s}: {value[metric_name]:0.4f}')


######################################################
# Multi-output models
# The models so far all predicted a single output feature, T (degC), for a single time step.

# All of these models can be converted to predict multiple features just by changing the number of units in the output layer and adjusting the training windows to include all features in the labels (example_labels)
######################################################

single_step_window = WindowGenerator(
    # `WindowGenerator` returns all features as labels if you
    # don't set the `label_columns` argument.
    train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=24, label_width=24, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}\n'
        f'Labels shape (batch, time, features): {example_labels.shape}')
# Note that the features axis of the labels now has the same depth as the inputs, instead of 1


baseline = Baseline()
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val, return_dict=True)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0, return_dict=True)

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])
history = compile_and_fit(dense, single_step_window)
val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])
history = compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate( wide_window.val, return_dict=True)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0, return_dict=True)

"""
Advanced: Residual connections
The Baseline model from earlier took advantage of the fact that the sequence doesn't change drastically from time step to time step. Every model trained in this tutorial so far was randomly initialized, and then had to learn that the output is a a small change from the previous time step.

While you can get around this issue with careful initialization, it's simpler to build this into the model structure.

It's common in time series analysis to build models that instead of predicting the next value, predict how the value will change in the next time step. Similarly, residual networks—or ResNets—in deep learning refer to architectures where each layer adds to the model's accumulating result.

Essentially, this initializes the model to match the Baseline. For this task it helps models converge faster, with slightly better performance.

This approach can be used in conjunction with any model discussed in this tutorial.

Here, it is being applied to the LSTM model, note the use of the tf.initializers.zeros to ensure that the initial predicted changes are small, and don't overpower the residual connection. There are no symmetry-breaking concerns for the gradients here, since the zeros are only used on the last layer.
"""

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta
  
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small.
        # Therefore, initialize the output layer with zeros.
        kernel_initializer=tf.initializers.zeros())
]))

history = compile_and_fit(residual_lstm, wide_window)
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val, return_dict=True)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0, return_dict=True)


## Multi-step into the future models
# Thus, unlike a single step model, where only a single future point is predicted, a multi-step model predicts a sequence of the future values.
# There are two rough approaches to this:
# Single shot predictions where the entire time series is predicted at once.
# Autoregressive predictions where the model only makes single step predictions and its output is fed back as its input.
# In the floowing section all the models will predict all the features across all output time steps.
OUT_STEPS = 24
multi_window = WindowGenerator(
   train_df=train_df, val_df=val_df, test_df=test_df,
   input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS)
multi_window


multi_val_performance = {}
multi_performance = {}

CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val, return_dict=True)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)



multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val, return_dict=True)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0, return_dict=True)

# to show all features: df.columns
multi_window.plot(multi_lstm_model); plt.show()
multi_window.plot(multi_lstm_model, plot_col='rh (%)'); plt.show()

#  Autoregressive model
class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the LSTM state.
  prediction, state = self.warmup(inputs)

  # Insert the first prediction.
  predictions.append(prediction)

  # Run the rest of the prediction steps.
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output.
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call

print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

history = compile_and_fit(feedback_model, multi_window)

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val, return_dict=True)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0, return_dict=True)


