
"""
built-in support for deep learning
C/C++ backend   faster than pure Python
support CPU, GPU and distributed cluster processing

struture based on data flow graph
computation units:
nodes = mathematical operations
edges = milti-dimentional array or tensor

keras is the default high level API for tensotflow 2.X
eager execution is the default for TensorFlow low level API.
TF 1.X -- tensorflow.python.framework.ops.Tensor
TF 2.X -- tensorflow.python.framework.ops.EagerTensor
"""
import tensorflow as tf
tf.executing_eagerly() # is True verify it is set to be eager execution 

# to disable 
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.executing_eagerly() # False

import numpy as np
a = tf.constant(np.array([1., 2., 3.]))
type(a)
# tensorflow.python.framework.ops.Tensor

"""
Typically, data in TensorFlow is packed into arrays where the outermost index is across examples (the "batch" dimension). The middle indices are the "time" or "space" (width, height) dimension(s). The innermost indices are the features.
"""

"""
Ways to build Neural Networks: Squential API and Functional API

The Sequential API is a simple way to build models layer by layer. Each layer has a fixed order, and data flows from one layer to the next.
The Functional API allows for more complex models, like multi-input,multi-output models, and models with shared layers.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

a = tf.constant([[1], [2], [3], [4]])
b = tf.constant([10, 20, 30])
print(a.shape, b.shape) # (4, 1) (3,)
result = a + b 
print(result.shape) # (4, 3) broadcasting

# fill the missing values in Tensorflow
# In Python nan can be created by: float('nan'), math.nan, np.nan
data = tf.constant([1.0, np.nan, 2.0])
filled_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
print(filled_data)

# tf.function is used to convert Python functions into TensorFlow computation graphs, improving performance
@tf.function
def add(a, b):
    return a + b
result = add(tf.constant(2), tf.constant(3))
print(result)




"""
Example: define the convolutional neural network where the convolution, pooling, and flattening layers will be applied. 
input shape of `(32, 32, 3)` because the images are of size 32 by 32, 3 color channels.
32 output filters, 3 by 3 feature detector, `same` padding to result in even padding for the input, `relu` activation function so as to achieve non-linearity, 
The next layer is a max-pooling layer
`pool_size` of (2, 2) define the pooling window, 2 strides,

Remember that you can design your network as you like. You just have to monitor the metrics and tweak the design and settle on the one that results in the best performance. In this case, another convolution and pooling layer is created. That is followed by the flatten layer whose results are passed to the dense layer. The final layer has 10 units because the dataset has 10 classes. Since it's a multiclass problem, the Softmax activation function is applied. 

tf.keras.layers.Conv2D()
data_format = "channels_last" or "channels_first",  you never set it, then it will be "channels_last".
If data_format="channels_last": A 4D tensor with shape: (batch_size, height, width, channels)
If data_format="channels_first": A 4D tensor with shape: (batch_size, channels, height, width)

Output shape:
If data_format="channels_last": A 4D tensor with shape: (batch_size, new_height, new_width, filters)
If data_format="channels_first": A 4D tensor with shape: (batch_size, filters, new_height, new_width)
Returns
A 4D tensor representing activation(conv2d(inputs, kernel) + bias).

the following example, the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). The width and height dimensions tend to shrink as you go deeper in the network. The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). Typically, as the width and height shrink, you can afford (computationally) to add more output channels in each Conv2D layer.

For a convolutional layer, the number of Parameters =
N_k * (K_w * K_h * C_i + 1)
Where N_k is the number of kernels or filters
K_w and K_h are the kernel width and height, respectively
C_i is the number of input channels
then, one additional bias term for each kernel/filter
For a fully connected layer, the number of parameters =
U_i * (U_o + 1)
where U_i is the number of input units
U_o is the number of output unit, 
one additional bias term for each output unit.

For batch normalization layers, each feature channel has two parameters (gamma and beta).
Parameters = 2 * num features (filters or channels)


"""
model = tf.keras.Sequential(
    [
    # the frist two parameters are filters and kernel_size
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
    # input image of shape (32, 32, 3), 32x32 pixal and RGB color_channels 
    # output shape (None, 32, 32, 32)  Param # 32*(3*3*3+1)   = 896
    # https://cs231n.github.io/convolutional-networks/
    # Output Shape is (None, 32, 32, 32) which shows (batch_size, height, width, channels)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    # Flattens the input. Does not affect the batch size.
    tf.keras.layers.Flatten(),
    # Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). units: Positive integer, dimensionality of the output space.
    tf.keras.layers.Dense(units=100, activation="relu"),
    # The Dropout layer helps prevent overfitting.
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=10, activation="softmax")
]
)

model.summary()

# plot the model structure and save it to a file
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
)

"""
Dropout regularization (layer). A specified percentage of connections are dropped during the training process.
This forces the network to learn patterns from the data instead of memorizing the data. This is what reduces overfitting. 
Dropout is implemented in Keras as a special layer type that randomly drops a percentage of neurons during the training process. When dropout is used in convolutional layers, it is usually used after the max pooling layer and has the effect of eliminating a percentage of neurons in the feature maps. When used after a fully connected layer, a percentage of neurons in the fully connected layer are dropped.
"""


""""
Compiling the model
The next step is to compile the model. The Sparse Categorical Cross-Entropy loss is used because the labels are not one-hot encoded. In the event that you want to encode the labels, then you will have to use the Categorical Cross-Entropy loss function. 

optimizer: 'adam', 'sgd', ...
loss: 'mse', ...

"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
How to halt training at the right time with Early Stopping
Left to train for more epochs than needed, your model will most likely overfit on the training set. One of the ways to avoid that is to stop the training process when the model stops improving. This is done by monitoring the loss or the accuracy. In order to achieve that, the Keras EarlyStopping callback is used. By default, the callback monitors the validation loss. Patience is the number of epochs to wait before stopping the training process if there is no improvement in the model loss. This callback will be used at the training stage. The callbacks should be passed as a list, even if it's just one callback.

How to save the best model automatically
You might also be interested in automatically saving the best model or model weights during training. That can be applied using a Keras ModelCheckpoint callback. The callback will save the best model after each epoch. You can instruct it to save the entire model or just the model weights. By default, it will save the models where the validation loss is minimum. 

"""
from tensorflow.keras.callbacks import EarlyStopping
callbacks = [
             EarlyStopping(patience=2)
]

checkpoint_filepath = '/tmp/checkpoint.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

callbacks = [
             EarlyStopping(patience=2),
             model_checkpoint_callback,
]

"""
Training the model
Let's now fit the data to the training set. The validation set is passed as well because the callback monitors the validation set. In this case, you can define many epochs but the training process will be stopped by the callback when the loss doesn’t improve after 2 epochs as declared in the EarlyStopping callback. 

"""
history = model.fit(X_train,y_train, epochs=600,validation_data=(X_test,y_test),callbacks=callbacks)

another_saved_model = tf.keras.models.load_model(checkpoint_filepath)

import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot
from tensorflow.keras.models import Model
print(tf.__version__)
#  only messages with severity "ERROR" or higher ("FATAL") will be displayed.
tf.get_logger().setLevel('ERROR')

"""
It is necessary to have both the model, and the data on the same device, either CPU or GPU, for the model to process data.

"/device:CPU:0": The CPU of your host machine, or "/device:GPU:<N>" the nth GPU on the host.
"/GPU:0": Short-hand notation for the first GPU of your machine that is visible to TensorFlow.
"/job:localhost/replica:0/task:0/device:GPU:1": Fully qualified name of the second GPU of your machine that is visible to TensorFlow.
By default, the GPU device is prioritized when the operation is assigned. if tf.config.list_physical_devices('GPU') returns an empty list, there is no GPU available on the host machine and all operations will be on CPU. Also when a TensorFlow operation has no corresponding GPU implementation, then the operation falls back to the CPU device. For example, tf.cast(x, dtype, name=None) only has a CPU kernel.


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

Manual device placement
you can use with tf.device to create a device context, and all the operations within that context will run on the same designated device.

import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.linalg.matmul(x, x)
  result = time.time()-start
  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

"""
# If shape is set, the value is reshaped to match. Scalars are expanded to fill the shape
tf.constant(0, shape=(2, 3))
tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

"""
tf.keras.layers.Dense(..., activation=None) 
torch.nn.Linear
They are now equal at this point. 
A linear transformation to the incoming data: y = x*W^T + b.
In PyTorch, we do
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(5, 30)
    def forward(self, state):
        return self.fc1(state)
or
trd = torch.nn.Linear(in_features = 3, out_features = 30)
y = trd(torch.ones(5, 3))
print(y.size())
# torch.Size([5, 30])
Its equivalent tf implementation would be
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, input_shape=(5,), activation=None)) 
(5,) is really (25,1) if the dimension is specified

tfd = tf.keras.layers.Dense(30, input_shape=(3,), activation=None)
x = tfd(tf.ones(shape=(5, 3)))
print(x.shape)
# (5, 30)

model.add (Dense(10, activation = None)) or nn.linear(128, 10) is the same, because it is not activated in both, therefore if you don't specify anything, no activation is applied.


TF Functional API: three steps, first define an input layer; define layers and connect each of them using Python functional syntax (whatis what gives the API it's name). Pyhton functional syntax is when you specify that the current layer is a function and the previous layer is a parameter to the funciton (when you define a layer you put the preceding layer in the parenthesis after the definition. may endup with double parenthesis syntax). Lastly you define the model by calling the model object and giving it the input and output layers.

"""
#  tf.keras.layers.Flatten() operation explained by keras.models.Model
import numpy as np
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(3,2,4))

# Define a model consisting only of the Flatten operation
# Flattens the input. Does not affect the batch size. C-like index order
prediction = tf.keras.layers.Flatten()(inputs)
prediction = tf.keras.layers.Dense(units=2)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=prediction)


X = np.arange(24).reshape(1,3,2,4)
print(X)
model.predict(X)

"""
tf.keras.layers.Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

Note: If the input to the layer has a rank greater than 2, Dense computes the dot product between the inputs and the kernel along the last axis of the inputs and axis 0 of the kernel (using tf.tensordot). For example, if input has dimensions (batch_size, d0, d1), then we create a kernel with shape (d1, units), and the kernel operates along axis 2 of the input, on every sub-tensor of shape (1, 1, d1) (there are batch_size * d0 such sub-tensors). The output in this case will have shape (batch_size, d0, units).

tf.keras.layers.Reshape(target_shape, **kwargs) where target_shape is a tuple of integers, does not include the samples dimension (batch size).
You can use -1 in the target_shape to let Keras infer the dimension size based on the total number of elements. For example, tf.keras.layers.Reshape(target_shape=(-1, 2)) will infer the first dimension based on the input size. where tf.keras.layers.Reshape(target_shape=(1, -1)) will infer the last dimension based on the input size. There might be None in some tensor shape values. The None element of the shape corresponds to a variable-sized dimension.
"""
######### Sequential model #########
# build a model with the Sequential API - list
seq_model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
     tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)]
     )
# or add method
seq_model = tf.keras.models.Sequential()
seq_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
seq_model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
seq_model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
seq_model.compile(optimizer='adam', loss='mse')

seq_model.summary()
seq_model.variables

######### Functional model #########
# build a model with the Functional API
input_layer = tf.keras.Input(shape=(28, 28))
# stack the layers using the syntax: new_layer()(previous_layer)
flatten_layer = tf.keras.layers.Flatten()(input_layer)
first_dense = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(flatten_layer)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(first_dense)
# declare inputs and outputs
func_model = Model(inputs=input_layer, outputs=output_layer)
func_model.compile(optimizer='adam', loss='mse')
func_model.summary()

# put model definitation and compilation into a function
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation = 'linear', input_dim = 784))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
    return model
model = get_model()
_ = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64,
              epochs=3, verbose=0, callbacks=[callback])
# Two ways to provide validation dataset to the fit() function
# model.fit(x=train_X, y=train_y, validation_data=(val_x, val_y)) # the default validation_data=None
# model.fit(x=X, y=y, validation_split=0.3) # the default validation_split=0.0, the fraction of the training data to be used as validation data. 

######### Subclassing: Provides the highest level of customization, allowing you to create custom layers and models #########

# If you need your custom layers to be serializable as part of a Functional model, you can optionally implement a get_config() method
# Note that the __init__() method of the base Layer class takes some keyword arguments, in particular a name and a dtype. It's good practice to pass these arguments to the parent class in __init__() and to include them in the layer config
# Best practice: deferring weight creation until the shape of the inputs is known. In the Keras API, we recommend creating layer weights in the build(self, inputs_shape) method of your layer. As  you may not know in advance the size of your inputs, and you would like to lazily create weights when that value becomes known, some time after instantiating the layer. 
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# Now you can recreate the layer from its config
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

X = Linear.from_config(config)
outputs = tf.keras.layers.Dense(units=1)(X)
model = tf.keras.Model(inputs=new_layer, outputs=outputs)


# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')
# model.summary()


# Build: defines the structure (blueprint) of the model. Usually, you do not need to all model.build() explicitly before training.
# Compile: configures the model for training by specifying the loss function, optimizer and metrics. 
# Train: Learns the optimal parameters for the model after the building.


############## Custom activation ##############


############## custom loss funciton ##############
model.compile(loss='mse', optimizer='sgd')
# the loss can either be decleared as a string or a loss object
from tensorflow.keras.losses import mean_squared_error
model.complie(loss=mean_squared_error, optimizer='sgd')
# or loss object with parameters, more flexibility for tuning hyperparameters. The loss funcitons expect two parameters, y_true and y_pred. Custom-built loss functions can leverage a wrapper function around the loss function with hyperparameters defined as its parameter. One other way of implementing a custom loss function is by creating a class with two function definitions, init and call.
model.complie(loss=mean_squared_error(param=value), optimizer='sgd')

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true, y_pred))

model.compile(loss=custom_loss, optimizer='adam')

############## Siamese Network ##############
# Siamese networks, often called twin networks, consist of a pair of neural networks that share their weights and aims at computing similarity functions.


############## Lambda Layer ##############
# You can either use lambda functions within the Lambda layer or define a custom function that the Lambda layer will call.
from tensorflow.keras import backend as K
def my_relu(x):
    return K.maximum(-0.1, x)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128),
  # use lambda functions within the Lambda layer
  # tf.keras.layers.Lambda(lambda x: tf.abs(x)), 
  # define a custom function that the Lambda layer will call
  tf.keras.layers.Lambda(my_relu), 
  tf.keras.layers.Dense(10, activation='softmax')
])

from tensorflow.keras import backend as K
"""
This line of code imports the backend module from TensorFlow's Keras API. The backend module provides various low-level functions and utilities that allow you to interact directly with the underlying TensorFlow computational graph. 

Access and manipulate tensors: You can use functions like K.get_value(), K.set_value(), and K.eval() to get and set the values of tensors.
Perform mathematical operations: The backend module provides a wide range of mathematical operations, such as K.sum(), K.mean(), K.max(), K.dot(), K.square(), K.sqrt(), etc., that you can apply to tensors.
Control the underlying TensorFlow session: You can use functions like K.get_session() to access the current TensorFlow session and K.clear_session() to clear it.
Set configuration options: You can use functions like K.set_floatx() to set the default floating-point precision used by Keras.
"""


############## custom layers ##############
import tensorflow as tf
from tensorflow.keras import backend as K

# define a custom activation function
def my_relu(x):
    return K.maximum(0.0, x)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(my_relu), 
    # these two lines of code are equivalent to 
    #tf.keras.layers.Dense(128, activation="relu")

    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# define a custom layer by subclassing tf.keras.layers.Layer
from tensorflow.keras.layers import Layer

class SimpleQuadratic(Layer):

    def __init__(self, units=32, activation=None):
        '''Initializes the class and sets up the internal variables'''
        super(SimpleQuadratic, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        '''Create the state of the layer (weights) This function is run when an istance is created: specifying the local input state and other necessary housekeepings. Here w is the weight and b is the biase'''
        a_init = tf.random_normal_initializer()
        a_init_val = a_init(shape=(input_shape[-1], self.units),
                            dtype='float32'
                           )
        # self.add_weight(
        self.a = tf.Variable(initial_value=a_init_val, trainable=True)
        
        b_init = tf.random_normal_initializer()
        b_init_val = b_init(shape=(input_shape[-1], self.units),
                            dtype='float32'
                           )
        self.b = tf.Variable(initial_value=b_init_val, trainable=True)
        
        c_init = tf.zeros_initializer()
        c_init_val = c_init(shape=(self.units,), dtype='float32'),
        self.c = tf.Variable(initial_value=c_init_val, trainable=True)
        super().build(input_shape) # ensures that your custom layer follows the standard Keras layer structure and behaves correctly within the framework.
   
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        x_squared = tf.math.square(inputs)
        return self.activation(tf.matmul(x_squared, self.a) + tf.matmul(inputs, self.b) + self.c)




############## Activation in Custom Layers ##############
class SimpleDense(Layer):

    # add an activation parameter
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)


    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
        super().build(input_shape)


    def call(self, inputs):
        # pass the computation to the activation layer
        return self.activation(tf.matmul(inputs, self.w) + self.b)
    
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    SimpleDense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

"""
tf.keras.Input(
    shape=None, batch_size=None, dtype=None,
    sparse=None, batch_shape=None, name=None, tensor=None
)
shape:	A shape tuple (tuple of integers or None objects), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; None elements represent dimensions where the shape is not known and may vary (e.g. sequence length).

Keras automatically adds the None value in the front of the shape of each layer, which is later replaced by the batch size.

When a popular kwarg input_shape is passed into tf_keras.layers.Dense(), then keras will create an input layer to insert before the current layer. This can be treated equivalent to explicitly defining an Input layer.
It is preferred to use an Input(shape) object as the first layer in the model instead.

input_shape=(None, features) or input_shape=(features, ) or input_shape=[features]
They are 
all accepted and the result is the same, but the tuples are preferred.

tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    ...)
Dense implements the operation: output = activation(dot(input, kernel) + bias)

units:	Positive integer, dimensionality of the output space, or the number of neurons in each layer of your neural network architecture
activation:	Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
Note: If the input to the layer has a rank greater than 2, Dense computes the dot product between the inputs and the kernel along the last axis of the inputs and axis 0 of the kernel (using tf.tensordot). For example, if input has dimensions (batch_size, d0, d1), then we create a kernel with shape (d1, units), and the kernel operates along axis 2 of the input, on every sub-tensor of shape (1, 1, d1) (there are batch_size * d0 such sub-tensors). The output in this case will have shape (batch_size, d0, units).

"""


# architectures with the functional API (vs sequential APIs)
# tf Keras API, tf.keras.layers.Input is used to instantiate a Keras tensor

# define inputs
input_a = tf.keras.Input(shape=[1], name="Wide_Input")
input_b = tf.keras.Input(shape=[1], name="Deep_Input")

# define deep path
# the deep input goes through dense layers whereas the wide/shallow go into the model directly

hidden_1 = tf.keras.layers.Dense(units=30, activation="relu")(input_b) # declears that this level should follow input_b
hidden_2 = tf.keras.layers.Dense(units=30, activation="relu")(hidden_1)

# define merged path
concat = tf.keras.layers.concatenate([input_a, hidden_2])
output = tf.keras.layers.Dense(units=1, name="Output")(concat)

# define another output for the deep path
aux_output = tf.keras.layers.Dense(units=1,name="aux_Output")(hidden_2)

# build the model
model = Model(inputs=[input_a, input_b], outputs=[output, aux_output])
model.summary()

# visualize the architecture
plot_model(model)


# from IPython.display import Image, display
# img = Image('model.png')  # Make sure the directory
# display(img)



# encapsulate the architecture into a class for tidy you code and easier reuse
# orchestrate multiple models in a solution

# Inheriting from the existing Model class lets you use the Model methods such as compile(), fit(), evaluate(), save(), save_weights(), summary() and keras.utils.plot_model().
class WideAndDeepModel(Model): # basic keras model class
    def __init__(self, units=30, activation='relu', **kwargs):
        '''initializes the instance attributes'''
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        '''defines the network architecture'''
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        
        return main_output, aux_output
    
# create an instance of the model
model = WideAndDeepModel()

# residual networks
# https://www.coursera.org/lecture/convolutional-neural-networks/resnets-HAhz9
# CNN residual type
class CNNResidual(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Conv2D(filters, (3, 3), activation="relu")
                     for _ in range(layers)]
        # len(layers) number of identical layers
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
           x = layer(x) # (x) is not argument instead specifies layer follows x
        # x will be a new stack of layers consisting of payer following x. this is the main path through the residual network block
        
        # concatenate the main path on the shortcut. The original input (residual) takes a shortcut or skip some connections to the end and combine with output from the layer (before into the activation function)
        return inputs + x

# similarly to construct a DNN residual 
class DNNResidual(Layer):
    def __init__(self, layers, neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(neurons, activation="relu")
                     for _ in range(layers)]
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
           x = layer(x)
        
        return inputs + x
    

class MyResidual(Model):
    # just define the layers, declear the states, no sequence
    def __init__(self, **kwargs):
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu")
        self.block1 = CNNResidual(2, 32)
        self.block2 = DNNResidual(2, 64) # this will be called 3 times
        self.out = tf.keras.layers.Dense(1)

    # how to construct the architecture
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.block1(x)
        for _ in range(1, 4): # this will run 3 times
            x = self.block2(x)
        return self.out(x)
    # Dense layer - CNNResiduala layer - 3 X DNNResidual layers -- Dense layer

############## Callbacks ##############
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
# tensorboard
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model.fit(train_batches, epochs=10, validation_data=validation_batches, 
          callbacks=[tensorboard_callback])
# model checkpoint
model.fit(train_batches, epochs=1, validation_data=validation_batches,verbose=2,
          # Callback to save the Keras model or model weights at some frequency.
          callbacks=[tf.keras.callbacks.ModelCheckpoint('saved_model', verbose=1)],
          # callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5',verbose=1),]
          # callbacks=[ModelCheckpoint('model.h5', verbose=1)]
           )
# early stopping
# A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing, considering the min_delta and patience if applicable. Once it's found no longer decreasing, model.stop_training is marked True and the training terminates. The quantity to be monitored needs to be available in logs dict. To make it so, pass the loss or metrics at model.compile().
model.compile(tf.keras.optimizers.SGD(), loss='mse')
model.fit(train_batches, epochs=50, validation_data=validation_batches,
          verbose=2, 
          # Stop training when a monitored metric has stopped improving.
          callbacks=[tf.keras.callbacks.EarlyStopping(
              patience=3, min_delta=0.05,  baseline=0.8, mode='min',monitor='val_loss', restore_best_weights=True, verbose=1)]
          )
# CSV logger
# Callback that streams epoch results to a CSV file.
# Each row in the CSV file corresponds to an epoch, and the columns represent various statistics such as the epoch number, training loss, training accuracy, validation loss, and validation accuracy.
filename = 'training.log'
csv_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=False)
model.fit(train_batches, epochs=5, validation_data=validation_batches,
          callbacks=[csv_logger]
          )
# learning rate scheduler
# Updates the learning rate during training based on predefined schedules or conditions.
def step_decay(epoch):
	initial_lr = 0.1 # Initial learning rate
	drop = 0.5 # Factor by which the learning rate will be reduced
	epochs_drop = 10 # Number of epochs after which the learning rate drops. the learning rate decrease every 10 epochs.
	lr = initial_lr * np.power(drop, np.floor((1+epoch)/epochs_drop))
	return lr
# (1 + epoch) calculates the current epoch number starting from 1 instead of 0.
# In this formula, np.floor((1 + epoch) / epochs_drop) calculates the number of times the learning rate has dropped so far. For example if epoch=20, then it is calculaated as np.floor((1+20)/10)=2. The learning rate will be 0.1*0.5**2=0.025

model.fit(train_batches, epochs=5, validation_data=validation_batches, 
          callbacks=[tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1),
                    tf.keras.callbacks.TensorBoard(log_dir='./log_dir')]
        )

# tf.keras.optimizers.schedules module to create various learning rate schedules

# Create an optimizer with the piecewise constant learning rate
optimizer = tf.keras.optimizers.Adam(
learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[1000, 2000], values=[0.1, 0.01, 0.001])
)
# In this example, the learning rate remains 0.1 until 1000 steps, drops to 0.01 between 1000 and 2000 steps, and further drops to 0.001 after 2000 steps.

# Create an optimizer with the exponential decay learning rate
optimizer = tf.keras.optimizers.Adam(
learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=0.1, decay_steps=1000, decay_rate=0.5,
staircase=True)
)

# reduceLROnPlateau
# ReduceLROnPlateau works by monitoring a specified metric (like validation loss) for a 'patience' number of epochs. If no improvement is seen in that metric for the specified number of epochs, it reduces the learning rate.
model.fit(train_batches, epochs=50, validation_data=validation_batches, 
          callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, verbose=1, patience=10,
            min_lr=0.0001),
            tf.keras.callbacks.TensorBoard(log_dir='./log_dir')]
        )


# custome callbacks
# Usage of `logs` dict. The `logs` dict contains the loss value, and all the metrics at the end of a batch or epoch. Example includes the loss and mean absolute error.
callback = tf.keras.callbacks.LambdaCallback(
on_epoch_end=lambda epoch,logs: 
    print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, logs["val_loss"] / logs["loss"]))
)

class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, ratio))

        if ratio > self.threshold:
            print("Stopping training...")
            self.model.stop_training = True

model = get_model()
_ = model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=64, epochs=3, verbose=0,
          callbacks=[callback]
          # callbacks=[DetectOverfittingCallback()]
          )





############## Convolutional neural network, ConvNet/CNN ##############
''' 
IMAGE_SIZE = (28, 28)
input_shape=IMAGE_SIZE + (3,)
input_shape is: (28, 28, 3)
input_shape=(3,) + IMAGE_SIZE
input_shape is: (3, 28, 28)
'''
import numpy as np
x = np.random.rand(4, 10, 10, 128)
y = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
print(y.shape)

def build_colv_model(dense_units, input_shape=IMAGE_SIZE + (3,)):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_units, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])
  return model
model = build_colv_model(dense_units=256)
model.compile(
    optimizer='sgd', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])



import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Layer

# an identity block is a standard block in a Residual Network (ResNet) that allows data to skip ahead by combining the output of early layers with the output of later layers. This process is called identity mapping. 

class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()
    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        # Use the Identity blocks that you just defined
        self.id1a = IdentityBlock(64, 3) # 64 filters of 3x3
        self.id1b = IdentityBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        # insert the identity blocks in the middle of the network
        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)
    
# utility function to normalize the images and return (image, label) pairs.
def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

# create a ResNet instance with 10 output units for MNIST
resnet = ResNet(10)
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir='./data')
dataset = dataset.map(preprocess).batch(32)

# train the model.
resnet.fit(dataset, epochs=1)


# Create named-variables dynamically
# Define a small class MyClass
class MyClass:
    def __init__(self):
        # One class variable 'a' is set to 1
        self.var1 = 1
        # vars(self)['var1'] = 1 # alternatively

# Create an object of type MyClass()
my_obj = MyClass()

# __dict__ is a Python dictionary that contains the object's instance variables and values as key value pairs.
my_obj.__dict__
# If you call vars() and pass in an object, it will call the object's __dict__ attribute, which is a Python dictionary containing the object's instance variables and their values as ke
vars(my_obj)

# Use a for loop to increment the index 'i'
for i in range(4,10):
    # Format a string that is var
    vars(my_obj)[f'var{i}'] = 0
    
# View the object's instance variables!
vars(my_obj)

# Format a string using f-string notation
i=1
print(f"var{i}")
# Format a string using .format notation
i=2
print("var{}".format(i))





"""
Gradient tapes
TensorFlow provides the tf.GradientTape API for automatic differentiation (a context manager); that is, computing the gradient of a computation with respect to some inputs, usually tf.Variables. TensorFlow "records" relevant operations executed inside the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape to compute the gradients of a "recorded" computation using reverse mode differentiation. This is particularly useful for custom training loops and gradient-based optimization.

tf.GradientTape can only track and record operations for those tensor variables, which are trainable, like tf.Variables.
For tracking operations of constant tensors (tf.constant), we need to tell the GradientTape to watch() the variable. This is because, constant tensors are not trainable by default.

GradientTape.gradient(target, sources, ...) method to computes the gradient (of the target against elements in source) using operations recorded in context of this tape.

w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

By default, tf.Variable(initial_value, trainable=True), if we set it to False, GradientTape will not track it just like a constant.

By default GradientTape will automatically watch any trainable variables that are accessed inside the context. If you want fine grained control over which variables are watched (specify with the .watch() method) you can disable automatic tracking by passing watch_accessed_variables=False to the tape constructor. By leveraging the tape watch method and watch_accessed_variables parameter, we have fine gained control over which variables are watched.

tf.GradientTape(persistent=False, watch_accessed_variables=True)

with tf.GradientTape(persistent=True) as tape:
  # if we want to track a constant, we need explicitly watch it by specifying
  # tape.watch(constant_name) 
  y = x @ w + b
  loss = tf.reduce_mean(y**2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])
We can then update the parameter using the learning rate and the gradients.

By default, the resources held by a GradientTape are released as soon as
GradientTape.gradient() method is called. To compute multiple gradients over
the same computation, create a persistent gradient tape (tf.GradientTape(persistent=True)). This allows multiple
calls to the gradient() method as resources are released when the tape object
is garbage collected.

With PyTorch, requires_grad=True parameter signals to torch.autograd engine that every operation on them should be tracked.
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
L = 3*a**3 - b**2

We can call .backward() on the loss function (L) of a and b, autograd calculates gradients of the L w.r.t parameters and store them in the respective's tensors' .grad attribute. For example,

external_grad = torch.tensor([1., 1.])
L.backward(gradient=external_grad)
# the gradient parameter specifies the gradient of the function being differentiated w.r.t. self. This argument can be omitted if self is a scalar. here we have a and b.
print(a.grad); print(9*a**2)
print(b.grad); print(-2*b)

"""
################# Custom Training Basics #################
# define a model
class Model(object):
  def __init__(self):
    # Initialize the weights to `2.0` and the bias to `1.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.w = tf.Variable(2.0)
    self.b = tf.Variable(1.0)

  def __call__(self, x):
    return self.w * x + self.b

model = Model()

# define a loss function
def loss(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))

# define a train loop
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dw, db = t.gradient(current_loss, [model.w, model.b])
  model.w.assign_sub(learning_rate * dw)
  model.b.assign_sub(learning_rate * db)

  return current_loss

# Finally, you can iteratively run through the training data and see how 'w' and 'b' evolve.
model = Model()

# Collect the history of W-values and b-values to plot later
list_w, list_b = [], []
epochs = range(15)
losses = []
for epoch in epochs:
  list_w.append(model.w.numpy())
  list_b.append(model.b.numpy())
  current_loss = train(model, xs, ys, learning_rate=0.1)
  losses.append(current_loss)
  print('Epoch %2d: w=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, list_w[-1], list_b[-1], current_loss))
  


################# Training Categorical #################

# define the model
def base_model():
  inputs = tf.keras.Input(shape=(784,), name='digits')
  x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

# optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# build training loop consisting of training and validation sequences
def apply_gradient(optimizer, model, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss_value = loss_object(y_true=y, y_pred=logits)
  
  gradients = tape.gradient(loss_value, model.trainable_weights)
  # optimizer.apply_gradients method applies calculated gradients to the model's trainable variables, updating them in the direction that minimizes the loss function.
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  
  return logits, loss_value

# 
def train_data_for_one_epoch():
  losses = []
  pbar = tqdm(total=len(list(enumerate(train))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
  for step, (x_batch_train, y_batch_train) in enumerate(train):
      logits, loss_value = apply_gradient(optimizer, model, x_batch_train, y_batch_train)
      
      losses.append(loss_value)
      
      train_acc_metric(y_batch_train, logits)
      pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
      pbar.update()
  return losses

def perform_validation():
  losses = []
  for x_val, y_val in test:
      val_logits = model(x_val)
      val_loss = loss_object(y_true=y_val, y_pred=val_logits)
      losses.append(val_loss)
      val_acc_metric(y_val, val_logits)
  return losses


# define the training loop that runs through the training samples repeatedly over a fixed number of epochs. Here you combine the functions you built earlier to establish the following flow:
# 1. Perform training over all batches of training data.
# 2. Get values of metrics.
# 3. Perform validation to calculate loss and update validation metrics on test data.
# 4. Reset the metrics at the end of epoch.
# 5. Display statistics at the end of each epoch.

model = base_model()

# Iterate over epochs.
epochs = 10
epochs_val_losses, epochs_train_losses = [], []
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))
  
  losses_train = train_data_for_one_epoch()
  train_acc = train_acc_metric.result()

  losses_val = perform_validation()
  val_acc = val_acc_metric.result()

  losses_train_mean = np.mean(losses_train)
  losses_val_mean = np.mean(losses_val)
  epochs_val_losses.append(losses_val_mean)
  epochs_train_losses.append(losses_train_mean)

  print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc)))
  
  train_acc_metric.reset_states()
  val_acc_metric.reset_states()

"""
Fundamentally, TensorFlow runs by means of computational graphs — i.e. a graph of nodes is used to represent a series of TensorFlow operations.
The core advantage of having a computational graph is allowing parallelism or dependency driving scheduling which makes training faster and more efficient.

tf.Graphs are data structures that contain tf.Operation and tf.Tensor.
tf.Operation objects represent units of computation.
tf.Tensor objects represent the units of data that flow between operations.
tf.Graphs lets your TensorFlow run fast, run in parallel, and run efficiently on multiple devices.
You create and run a graph in TensorFlow by using tf.function, either as a direct call (e.g. tf.function(a_regular_function)) or as a decorator to a regular function.
tf.function uses a library called AutoGraph (tf.autograph) to convert Python code into graph-generating code.
tf.function applies to a function and all other functions it calls.

The purpose of AutoGraph is to graph code through the transformation of code written in Python's classic syntax structure into TensorFlow graph-compatible code. 

You can use @tf.function decorator to make graphs out of your programs. It is a transformation tool that creates Python-independent dataflow graphs out of your Python code. This will help you create performant and portable models, and it is required to use SavedModel.

The main takeaways and recommendations are:
Debug in eager mode, then decorate with @tf.function.
Don't rely on Python side effects like object mutation or list appends.
tf.function works best with TensorFlow ops; NumPy and Python calls are converted to constants.

tf.function best practices
1. Toggle between eager and graph execution early and often with tf.config.run_functions_eagerly to pinpoint if/ when the two modes diverge.
2. Create tf.Variables outside the Python function and modify them on the inside. The same goes for objects that use tf.Variable, like tf.keras.layers, tf.keras.Models and tf.keras.optimizers. The graph mode function should only contain operations on tensors (as arguments in function defination)
3. Avoid writing functions that depend on outer Python variables, excluding tf.Variables and Keras objects.
4. Prefer to write functions which take tensors and other TensorFlow types as input. You can pass in other object types but be careful! Learn more in Depending on Python objects of the tf.function guide.
5. Include as much computation as possible under a tf.function to maximize the performance gain. For example, decorate a whole training step or the entire training loop.

"""
# define the variables outside of the decorated function
v = tf.Variable(1.0)

@tf.function
def a_regular_function(x):
    ...
    print()
    tf.print() # differences?
    return v.assign_add(x)

# to show how AutoGraph has transformed the original Python code into TensorFlow graph-compatible code
print(tf.autograph.to_code(a_regular_function.python_function))

""""

"""
#  Basic mirrored strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {strategy.num_replicas_in_sync}')
# if you run on a single device it might take longer because of overhead in implementing the strategy. The advantages of using this strategy is more evident if you use it on multiple devices. 
with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
    )
model.fit(train_dataset, epochs=12)

# Multi-GPU mirrored strategy
# Note that it generally has a minimum of 8 cores. This GPU has 4 cores 
import tensorflow as tf
import numpy as np
import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
# If you have *different* GPUs in your system, you probably have to set up cross_device_ops like this
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print ('Number of devices: {strategy.num_replicas_in_sync}')





