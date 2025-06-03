
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



# Ablone dataset example
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()
abalone_train.info()

abalone_train.describe().transpose()

abalone_features = abalone_train.copy()

"""
In TensorFlow, you don't strictly need to convert both the predictor (features) and response (labels) to NumPy arrays. Numpy Arrays (direct usable), Tensorflow Tensors (direct usable), Pandas DataFrame (converted internally) and Tensorflow Datasets (tf.data.Dataset) are all acceptable.
Best Practice: Convert Pandas DataFrames and Series to NumPy Arrays or Tensors.

"""
# pop the response variable from the predictors
abalone_labels = abalone_features.pop('Age').to_numpy()
# treat all features identically and pack them into a single NumPy array
abalone_features = np.array(abalone_features)



# define empty pd.DataFrame for initialiation
df = pd.DataFrame()
train_df = val_df = test_df = df

def add_to_class(Class):
    """Set the value of the named attribute in created class by using Python built-in function setattr(object, name, value) """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class WindowGenerator():
    def __init__(self, input_width=1, label_width=1, shift=1,
                 train_df=train_df, val_df=val_df, test_df=test_df,label_columns=None):
        """
         input_width: int
            The width of the input window (time-step)
        label_width: int
            The width of the label window
        shift: int
            The number of units shifted to the end of the input and label windows. The offset or the number of time units into the future. 
        train_df, val_df and test_df must already exist in the namespace and you do not need to pass these arguments when instantiating the class.
        """

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # Feature column indices
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # slice(start=0, end, step=1)
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None) # None specifies all the way to the end
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """"
        Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels or (features, labels) pairs.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            # Stacks a list of rank-R tensors into a rank-(R+1) tensor, equivalent to np.stack()
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes manually. This way the `tf.data.Datasets` are easier to inspect.
        # tf.ensure_shape() is prefered over tf.set_shape() but here AttributeError: 'SymbolicTensor' object has no attribute 'ensure_shape'
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data):
        """ Take a time series DataFrame and convert it to a tf.data.Dataset of (input_window, label_window) if targets was passed """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        
        return ds.map(self.split_window)
    
    @property
    def train(self):
       return self.make_dataset(self.train_df)
    @property
    def val(self):
       return self.make_dataset(self.val_df)
    @property
    def test(self):
       return self.make_dataset(self.test_df)
    @property
    def example(self):
       """Get and cache an example batch of `inputs, labels`."""
       result = getattr(self, '_example', None)
       if result is None:
           # No example batch was found, so get one from the `.train` dataset
           result = next(iter(self.train))
       # And cache it for next time
       self._example = result
       return result
    
    def __repr__(self): # return a printable representation of the object
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Shift/offset: {self.shift}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

@add_to_class(WindowGenerator)
def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')




######### Some data preparation methods #########
import tensorflow_datasets as tfds
path = "./tensorflow_datasets"
(train_examples, validation_examples, test_examples), info = tfds.load('horses_or_humans', data_dir=path, as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'])
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
SIZE = 150 #@param {type:"slider", min:64, max:300, step:1}
IMAGE_SIZE = (SIZE, SIZE)
BATCH_SIZE = 32 #@param {type:"integer"}
def format_image(image, label):
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0
  return  image, label
train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
# prefetch one batch of data into CPU and make sure there is always one ready while the GPU is working on forward/backward propagation on the current batch
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)




# numerical data
from sklearn.model_selection import train_test_split
train, test = train_test_split(input_df, test_size=0.2)

def norm(train_x):
    '''Normalize the data using the descriptive statistics'''
    train_stats = train_x.describe() # descriptive statistics.
    return (train_x - train_stats['mean']) / train_stats['std']

def format_output(data: pd.DataFrame, target:str):
    y = data.pop(target).to_numpy() # or
    # y = np.array(input.pop(target))
    return y

# get the target out and format the train/test sets
train_Y = format_output(train)
test_Y = format_output(test)
# Normalize the training and test data
norm_train_X = norm(train)
norm_test_X = norm(test)

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
              # when you have two response variables
              loss={'y1_output': 'mse',
                    'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})
history = model.fit(norm_train_X, train_Y,
                    epochs=500, batch_size=10, validation_data=(norm_test_X, test_Y))
# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
Y_pred = model.predict(norm_test_X)

def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()
plot_metrics(metric_name='y1_output_root_mean_squared_error', title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2_output_root_mean_squared_error', title='Y2 RMSE', ylim=7)

# prepare fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
# configure, train, and evaluate the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)




# shuffle, batch and prefetch

import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
shuffled_dataset = dataset.shuffle(buffer_size=3)

for element in shuffled_dataset:
  print(element) 


dataset = tf.data.Dataset.range(9)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
list(dataset.as_numpy_iterator())
list(dataset.as_numpy_iterator())

dataset = tf.data.Dataset.range(9)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
list(dataset.as_numpy_iterator())
list(dataset.as_numpy_iterator())

dataset = tf.data.Dataset.from_tensor_slices([0, 0, 0, 1, 1, 1, 2, 2, 2])
dataset = dataset.batch(3)
list(dataset.as_numpy_iterator())

######### example ##########
# Batch the input data
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Create Datasets from the batches
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
# almost always we do shuffle before batch to avoid overfit to whatever structure was in the input data. Note we do not need to shuffle the test set.


a = tf.constant(1.0, dtype=tf.float64)
b = tf.constant([1.0, 2.0], dtype=tf.float64)

a.shape
b.shape

rank_4_tensor = tf.zeros([3, 2, 4, 5])
rank_4_tensor.shape # Shape: The length (number of elements) of each of the axes of a tensor
rank_4_tensor.shape[0]
rank_4_tensor.ndim  # Axis or Dimension
rank_4_tensor.dtype
tf.size(rank_4_tensor).numpy() # Size: The total number of items in the tensor

# or if you need a tensor output 
tf.rank(rank_4_tensor)
tf.shape(rank_4_tensor)
# While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.
tf.debugging.set_log_device_placement(True)
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
