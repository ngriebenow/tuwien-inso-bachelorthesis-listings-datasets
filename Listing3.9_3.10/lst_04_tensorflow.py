import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# model weight, bias and function
W = tf.Variable(np.random.normal())
b = tf.Variable(np.random.normal())
def f(x):
    return W * x + b
print(f"W = {W.numpy():.4f}, b = {b.numpy():.4f}")

# generate synthetic test data (y,W) with actual weight, bias and some noise
act_W = 4.7
act_b = -0.3

NUM_EXAMPLES = 1000
X = tf.random.normal(shape=(NUM_EXAMPLES,))
noise = tf.random.normal(shape=(NUM_EXAMPLES,))
y = X * act_W + act_b + noise

epochs = 12
learn_rate = 0.1

for e in range(epochs):

    # mean of squares as loss function
    with tf.GradientTape() as t:
        current_loss = tf.reduce_mean(tf.square(y - f(X)))
    
    # calculate gradients
    delta_W, delta_b = t.gradient(current_loss, [W, b])
    W.assign_sub(learn_rate * delta_W)
    b.assign_sub(learn_rate * delta_b)

    print(f"W = {W.numpy():.4f}, b = {b.numpy():.4f} loss = {current_loss.numpy():.4f} in epoch {e}")