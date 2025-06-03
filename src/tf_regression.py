
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras import layers
print(tf.__version__)

print(f"Last run: {datetime.datetime.now()}")

# Ablone dataset
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()
abalone_train.info()

abalone_train.describe().transpose()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age').to_numpy()

# treat all features identically and pack them into a single NumPy array
abalone_features = np.array(abalone_features)

# df = abalone_train.copy()
# columns_to_pop = ['Age', 'Shell weight']
# pd.DataFrame({col: df.pop(col) for col in columns_to_pop})

### basic simple linear regression model
# only a a single input tensor, Sequential model is sufficient
abalone_model = tf.keras.Sequential([
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(1)
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)

### use the normalization layer in your model
# a preprocessing layer that normalizes continuous features.
normalize = keras.layers.Normalization()
# adapt() method learn layer state directly from the input data
normalize.adapt(abalone_features)
norm_abalone_model = tf.keras.Sequential([
  normalize,
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)

###  Model on dateset of the different data types and ranges
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)
# Perform a calculation using the input
result = 2*input + 1
# the result doesn't have a value, just a representation of the calculation or graph mode instead of eager mode (which must has values)
result

calc = tf.keras.Model(inputs=input, outputs=result)

print(calc(np.array([1])).numpy())
print(calc(np.array([2])).numpy())


# build a set of symbolic tf.keras.Input objects, matching the names and data-types of the CSV columns.
inputs = {}
for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs

# concatenate the numeric inputs together and run them through a normalization layer
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = keras.layers.Concatenate()(list(numeric_inputs.values())) # Functional interface
# some example has the list of input tensors inside Concatenate()

# a preprocessing layer that normalizes continuous features.
norm = keras.layers.Normalization()
# adapt() method learn layer state directly from the input data
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

# Collect all the symbolic preprocessing results
preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = keras.layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs_cat = keras.layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
  ])
  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model

"""
class LogisticRegression(tf.keras.Model):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(num_features,))

    def call(self, inputs):
        return self.linear(inputs)

num_features = X_train.shape[1]
model = LogisticRegression(num_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history =  model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
"""


titanic_model = titanic_model(titanic_preprocessing, inputs)

titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

# titanic_model.save('test.keras')
# reloaded = tf.keras.models.load_model('test.keras')

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_model(features_dict)




################
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = keras.layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = keras.layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a 'tf.data.Dataset' that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

"""
# A preprocessing layer that maps strings to (possibly encoded) indices.
tf.keras.layers.StringLookup(
    max_tokens=None,vocabulary=None,output_mode='int',...
)
# When output_mode is "int", the vocabulary will begin with the mask token (if set), followed by OOV indices, followed by the rest of the vocabulary. When output_mode is "multi_hot", "count", or "tf_idf" the vocabulary will begin with OOV indices and instances of the mask token will be dropped.

# A preprocessing layer that maps integers to (possibly encoded) indices.
tf.keras.layers.IntegerLookup(
    max_tokens=None,vocabulary=None,vocabulary_dtype='int64',output_mode='int', ...
)
The vocabulary for the layer must be either supplied on construction or learned via adapt(). During adapt(), the layer will analyze a data set, determine the frequency of individual integer tokens, and create a vocabulary from them. 

tf.keras.layers.CategoryEncoding(
    num_tokens=None, output_mode='multi_hot', sparse=False, **kwargs
)

layer = keras.layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
layer([3, 2, 0, 1])

layer = keras.layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
layer([[0, 1], [0, 0], [1, 2], [3, 1]])

"""
# multi-hot encode the variables with integer categorical values
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')

encoding_layer = get_category_encoding_layer(name='Age',
                                             dataset=train_ds,
                                             dtype='int64',
                                             max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs['Age'] = age_col
encoded_features.append(encoded_age_col)

# multi-hot encode the variables with string categorical values
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']

for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs[header] = categorical_col
  encoded_features.append(encoded_categorical_col)







url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy().dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = keras.layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)


linear_model = tf.keras.Sequential([
    normalizer,
    # performs y=mx + b where m is a matrix and x is a vector
    keras.layers.Dense(units=1)
])
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

linear_model.summary()

linear_model.predict(train_features[:5]) # built—check the kernel weight size maetches the data size
linear_model.layers[1].kernel

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)





from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf


# 5 LSTM layers. This is called Stacked/Deep LSTM instead of just 1 LSTM layer (called Simple LSTM) for better forecast.
model = Sequential()
# Use "return_sequences = True" if using multiple LSTM layers, except in the last LSTM layer (otherwise we will face dimension mismatch error)
model.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
# Dropout regularization layers, amount of neurons to ignore in the layers
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) # one dimensional real output
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 120, batch_size = 6)


from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam 



n_features = 1                        
n_input = 10

# Stacked/Deep LSTM wiht more than one LSTM layers
model1 = Sequential()
model1.add(InputLayer((n_input,n_features)))
# return_sequences:	Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
model1.add(LSTM(units=100, return_sequences = True, name='LSTM1'))     
model1.add(LSTM(units=100, return_sequences = True, name='LSTM2'))
model1.add(LSTM(units=50, name='LSTM3'))
model1.add(Dense(units=8, activation = 'relu', name='Dense1'))
model1.add(Dense(units=1, activation = 'linear', name='Dense-Out'))
model1.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
model1.compile(loss = MeanSquaredError(), 
               optimizer = Adam(learning_rate = 0.0001), 
               metrics = RootMeanSquaredError())
model1.fit(X_train, y_train, 
           validation_data = (X_val, y_val), 
           epochs = 50, 
           callbacks = [early_stop])

# save the model
save_model(model1, "LSTM_Models/lstm_univariate.h5")
# load the model
model1 = load_model('LSTM_Models/lstm_univariate.h5')