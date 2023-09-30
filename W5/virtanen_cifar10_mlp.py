import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

EPOCHS = 10
LEARNING_RATE = 0.08 

(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()

# flatten the images to single vector
x = x.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

# then normalize
x = x.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# convert integer values to one-hot vectors
y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, input_dim=3072, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=LEARNING_RATE)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

tr_hist = model.fit(x, y, epochs=EPOCHS, batch_size=100, verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"\nclassification accuracy for test data:")
print(f"loss: {test_loss} - accuracy: {test_accuracy}")



plt.plot(tr_hist.history['loss'], label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training'], loc='upper right')
plt.show()
