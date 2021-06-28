#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

true_w = tf.constant([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 10
data_iter = load_array((features, labels), batch_size)

initializer = tf.initializers.RandomNormal(stddef=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))

#Loss Function and optimizer
loss = tf.keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.SGD(eta=0.03)

#Number of iterations
iter = 3

for i in range(iter):
    for X,y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X,training=True),y)
        grads = tape.gradient(l,net.trainable_variables)
        optim.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
