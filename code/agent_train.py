import tensorflow as tf
from agent import model
import datetime
import time
import sys
import os
import numpy as np

# ============ loading the data with TF data API ==============
### FROM NUMPY
"""
# Load the training data into two NumPy arrays, for example using `np.load()`.
#with np.load("/var/data/training_data.npy") as data:
#  features = data["features"]
#  labels = data["labels"]
#dataset = tf.data.Dataset.from_tensor_slices((features, labels))
"""

### FROM CSV 
"""
#filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
#record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
#dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4])
"""

### Creating the batches
"""
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
dataset = dataset.shuffle(buffer_size=10000)
batched_dataset = dataset.batch(4)
dataset = dataset.repeat() #for multiple epochs 

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
"""


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# ==================== Parameters =====================================

# Model Hyperparameters
tf.flags.DEFINE_integer("param1", 32, "parameter 1 (default: 32)")
tf.flags.DEFINE_integer("param2", 64, "parameter 2 (default: 64)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 2000)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("run_name", None, "Suffix for output directory. If None, a timestamp is used instead")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# ==================== Data Loading and Preparation ===================

# downloads the MNIST data if it doesn't exist
# each image is of size 28x28, and is stored in a flattened version (size 784)
# the label for each image is a one-hot vector (size 10)
# the data is divided in training set (mnist.train) of size 55,000, validation set
# (mnist.validation) of size 5,000 and test set (mnist.test) of size 10,000
# for each set the images and labels are given (e.g. mnist.train.images of size
# [55,000, 784] and mnist.train.labels of size [55,000, 10])
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ================= Define the session graph =====================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    # Create a session
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Output directory for models and summaries
        if FLAGS.run_name is None:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        else:
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.run_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. TF assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()

        # Build the graph
        model = model(param1=FLAGS.param1, param2=FLAGS.param2)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(model.loss)  # available var_list = <var to optimize>

        # Or with gradient clipping
        # grads_and_vars = optimizer.compute_gradients(model.loss)
        # clipped_grads_and_vars = [(tf.clip_by_norm(grad, <clipping threshold>), var) for grad, var in grads_and_vars]
        # train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Example to add a summary of a variable not computed directly in the graph
        # step_time_placeholder = tf.placeholder(dtype=tf.float32, shape={})
        # step_time_summary = tf.summary.scalar("step_time", step_time_placeholder)
        # last_step_time = 0.0

        # Example to print one cnn kernel
        # V = tf.slice(cnn.W_conv1, (0, 0, 0, 0), (-1, -1, -1, 1))
        # V = tf.reshape(V, (-1, 5, 5, 1))
        # image_summary_op = tf.summary.image("kernel_layer1", V, 5)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])  # add the others summaries if needed
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Initialize all the variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
                A single training step
                """
            feed_dict = {
                model.x: x_batch,
                model.y_true: y_batch,
                # add other placeholders here
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            train_summary_writer.flush()


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.x: x_batch,
                model.y_true: y_batch,
                # add other placeholders here
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        for i in range(FLAGS.num_epochs):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            begin = time.time()
            train_step(batch[0], batch[1])
            end = time.time()
            last_step_time = end - begin

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(validation.images, validation.labels, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        # after the training print the test accuracy. 
        # This can also be done in a separate test.py file - depending on what is need to do.
        print("test accuracy %g" % model.accuracy.eval(feed_dict={
            model.x: test.images, model.y_true: test.labels}))
