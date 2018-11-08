import tensorflow as tf

# Model Hyperparameters
tf.flags.DEFINE_integer("hidden_units", 900, "number of hidden units for each layer (default: 900)")
tf.flags.DEFINE_integer("n_layers", 5, "number of layers (default: 5)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning rate (default: 0.0001)")

class model(object):
    def __init__(self, param1, param2):
        # These variables are used to feed data into the graph
        self.x = tf.placeholder(name='x', shape=(None, 27), dtype=tf.int32)
        self.labels = tf.placeholder(name='labels', shape=None, dtype=tf.int32)

        # Create a list of layer sizes for our network
        layer_sizes = [2] + [hidden_units] * (n_layers - 1)

        # For each layer except the last define an affine transformation followed by a nonlinearity
        layer_output = self.x
        nonlinearity = tf.nn.relu

        for i, (in_size, out_size) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            with tf.variable_scope('layer' + str(i), reuse=None):
                W = tf.get_variable(name='W' + str(i), shape=(in_size, out_size), dtype=tf.float32)
                b = tf.get_variable(name='b' + str(i), shape=out_size, dtype=tf.float32)
                layer_output = nonlinearity(tf.matmul(layer_output, W) + b)

        # Compute two logits for a softmax layer
        with tf.variable_scope('softmax', reuse=None):
            softmax_W = tf.get_variable(name='softmax_W', shape=(hidden_units, 2), dtype=tf.float32)
            softmax_b = tf.get_variable(name='softmax_b', shape=2, dtype=tf.float32)
            logits = tf.matmul(layer_output, softmax_W) + softmax_b

        # Instead of explicitly computing softmax, just pass logits to this loss functions
        # It will one loss per example so take the mean over the batch
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels))

        # Predictions are not used during training but are used for evaluation
        self.predictions = tf.argmax(logits, 1)

        # This node is used to apply gradient information to modify variables
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)