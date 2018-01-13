# Loss functions
# -------------------------
#
# This python script illustrates the different
# loss functions for regression and classification

import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.python.framework import ops 
ops.reset_default_graph()

sess = tf.Session() 

### Numerical Predictions
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# L2 loss
# L = (pred - actual)^2
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# L1 loss
# L = abs(pred - actual)
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Huber loss
# L = delta^2 * (sqrt(1 + ((pred -actual)/delta)^2)-1)
delta = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2))-1.)
phuber2_y_out = sess.run(phuber2_y_vals)

# Plot the output:
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()


##### Categorical Predictions #####
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

# Hinge loss
# Use for predicting binary (-1, 1) classes
# L = max(0, 1-(pred*actual))
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

# Cross entropy loss
# L = -acutal * (log(pred)) - (1-actual)(log(1-pred))
xentropy_y_vals = -tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

# L= -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
# or
# L = max(actual, 0) - actual*pred + log(1+exp(-abs(actual)))
