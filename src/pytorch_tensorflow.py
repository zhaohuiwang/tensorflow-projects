
"""
################################################################################
TensorFlow provides an easy way to build, compile, and train a model. It’s highly optimized for deployment and production scenarios. The API is mature and widely supported across various platforms.

TensorFlow Pros:
    Great for production environments
    Powerful ecosystem (TensorFlow Lite, TensorFlow Serving)
    Built-in tools for visualization (TensorBoard)
TensorFlow Cons:
    Steeper learning curve for beginners
    Verbose syntax at times
################################################################################
"""
import tensorflow as tf

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5)

""""
################################################################################
PyTorch is loved by researchers and is often praised for its dynamic computational graph and ease of use.

PyTorch Pros:
    Easier to debug due to dynamic computation graph
    Great for research and prototyping
    Simpler, more intuitive syntax
PyTorch Cons:
    Lacks the same level of production support as TensorFlow (though it's improving)
    Fewer pre-built tools for deployment

################################################################################
"""
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = SimpleNN()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(5):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()

"""
################################################################################
Keras is an open-source high-level neural networks API written in Python.
The MNIST dataset is one of the most famous datasets in machine learning. It contains 70,000 images of handwritten digits (0-9). Each image is a grayscale picture, 28x28 pixels in size. The goal is to classify these images into one of the ten digit categories.

################################################################################"""    
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Flatten the 28x28 images into a 1D vector of 784 pixels
    layers.Dense(128, activation='relu'),      # Add a fully-connected (Dense) layer with 128 neurons
    layers.Dense(10, activation='softmax')     # Output layer with 10 neurons (one for each digit 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_acc}')


