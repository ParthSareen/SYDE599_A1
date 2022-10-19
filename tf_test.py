import tensorflow as tf
import numpy as np
import pickle

with open('datafiles/assignment-one-test-parameters.pkl', 'rb') as f:
    data = pickle.load(f)

weights_layer_one = data['w1']
weights_layer_two = data['w2']
weights_layer_three = data['w3']
bias_layer_one = data['b1']
bias_layer_two = data['b2']
bias_layer_three = data['b3']
inputs = data['inputs']
targets = data['targets']

i = tf.keras.Input(shape=2)
x = tf.keras.layers.Dense(10)(i)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(10)(x)
x = tf.keras.layers.ReLU()(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=i, outputs=output)

model.layers[1].set_weights((np.transpose(np.array(weights_layer_one)), bias_layer_one))
model.layers[3].set_weights((np.transpose(np.array(weights_layer_two)), bias_layer_two))
model.layers[5].set_weights((np.transpose(np.array(weights_layer_three)), bias_layer_three))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())

pred = model.predict(inputs)
truth = np.expand_dims(np.array(targets), -1)

# print((pred - truth) * (pred - truth)/ 2)

model.fit(inputs, targets, batch_size=200, epochs=5)
print(model.predict(inputs))
# print(model.layers[1].get_weights())
