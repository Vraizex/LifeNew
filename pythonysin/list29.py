import tensorflow as tf
sess = tf.InteractiveSession()

raw_data = [1., 2., 3., 4., 5., 6.,]

spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-5] >5:
        updater = tf.assign()