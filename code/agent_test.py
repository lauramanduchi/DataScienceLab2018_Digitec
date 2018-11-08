
import tensorflow as tf
import numpy as np
import os
import time
import datetime
		
## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "/data/sentences.eval", "Path to the test data. This data should be distinct from the training data.")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523379413/checkpoints/", "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")

## DATA PREPARATION ##

# Load data
print("Loading and preprocessing test dataset \n")
#x_test, y_test = preprocessing.preprocess(FLAGS.data_file_path, preprocessing.find_vocabulary("/data/sentences.train")) #would be more intelligent to save the vocab

## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		
        # Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		# Get the placeholders from the graph by name
		input = graph.get_operation_by_name("input").outputs[0]

		# Tensors we want to evaluate
		probas = graph.get_operation_by_name("model/softmax").outputs[0]

		# Generate batches for one epoch
		# batches = preprocessing.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

		# Collect the predictions here
		all_probas = []

		for x_test_batch in batches:
			batch_probas = sess.run(probas, {input_x: x_test_batch})
			all_probas = np.concatenate([all_predictions, batch_predictions])

print("Total number of test examples: {}".format(len(y_test))) #to remove
# or 
#print("test accuracy %g"%model.accuracy.eval(feed_dict={
#            model.x: test.images, model.y_true: test.labels}))