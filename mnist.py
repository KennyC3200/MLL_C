import tensorflow_datasets as tfds
import numpy as np

def ds_to_numpy(ds):
    return np.array([n[0].numpy() for n in ds]), np.array([n[1].numpy() for n in ds])

# Training dataset is where the number/label matches a perfect image
# E.g. 7 in the label will map to the perfect image of 7
# Testing dataset is where the number/label matches an imperfect image
# E.g. 7 in the label will map to an imperfect image of 7 which kind of looks like a 1 and the
# neural network needs to decide between 1 or 7
train_ds, test_ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)

train_images, train_labels = ds_to_numpy(train_ds)
test_images, test_labels = ds_to_numpy(test_ds)

# The image pixel format is [0, 255] but ML works better for [0.0, 1.0]
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# These labels don't use one-hot encoding
# Transforming the labels into one-hot encoding will be done in C
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.tofile("train_images.mat")
train_labels.tofile("train_labels.mat")
test_images.tofile("test_images.mat")
test_labels.tofile("test_labels.mat")

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)