import tensorflow as tf

random_tensor = tf.random.uniform((10,10), minval=0, maxval=1.0)
print(random_tensor)
salt_mask = tf.cast(random_tensor <= (0.06 / 2), random_tensor.dtype)
print(salt_mask)
pepper_mask = tf.cast(random_tensor >= (1 - 0.06 / 2), random_tensor.dtype)
print(pepper_mask)