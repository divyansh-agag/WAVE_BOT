import tensorflow as tf
import numpy as np
train_x=np.load("train_x.npy")
train_y=np.load("train_y.npy")

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
# Define the model
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(21, activation='relu'),                    # another hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')                  # output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=10)

# Predict on a new point
model.save("wave_bot.h5")
