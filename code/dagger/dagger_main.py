import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tensorflow import keras
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt 

import dagger.dagger_utils as dagger_utils
from utils.init_dataframes import init_df
from utils.load_utils import load_obj, save_obj
import utils.sampler as sampler
from dagger.model import create_model
import utils.algo_utils

# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)


""" 
This module runs DAgger Algorithm
    1) Get initial dataset of trajectories from the teacher
    2) Train a policy that best mimics the expert on those trajectories
    3) At each iteration, collect more trajectories (teacher's output state given previous states in network) 
       and adds those trajectories to the dataset
    4) Train the network again to find the policy that best mimics the teacher on the aggregated dataset

The results file can be found in the runs/cp.ckpt folder. 
"""
# ============= PARAMETERS ========== #

# Data loading parameters
tf.flags.DEFINE_string("run_name", None, "Run name to save (default: none)")
tf.flags.DEFINE_integer("threshold", 50, "Length of the final subset of products (default: 50")
tf.flags.DEFINE_integer("in_maxMI_size", 1000, "Initial number of products to run maxMI algorithm on (default:1000)")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_float("val_split", 0.1, "Fraction used for validation during training (default: 0.1)")
tf.flags.DEFINE_integer("n_epochs", 50, "Number of epochs during secondary training")
tf.flags.DEFINE_integer("n_epochs_init", 100, "Number of init epochs (default: 1000)")
tf.flags.DEFINE_integer("n_episodes", 3000, "Number of episodes (default: 500)")
tf.flags.DEFINE_integer("h1", 256, "Number of hidden units (default: 256)")
tf.flags.DEFINE_integer("h2", 128, "Number of hidden units layer 2 (default: 128)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "checkpoint every")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
"""Printing model configuration to command line"""
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

# Loading the datasets
try:
    products_cat = load_obj('../data/products_table')
    traffic_cat = load_obj('../data/traffic_table')
    purchased_cat = load_obj('../data/purchased_table')
    filters_def_dict = load_obj('../data/filters_def_dict')
    type_filters = load_obj('../data/type_filters')
    question_text_df = load_obj('../data/question_text_df')
    answer_text = load_obj('../data/answer_text')
    print("Loaded datasets")
except:
    print("Creating datasets...")
    products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text = init_df()
    save_obj(products_cat, '../data/products_table')
    save_obj(traffic_cat, '../data/traffic_table')
    save_obj(purchased_cat, '../data/purchased_table')
    save_obj(type_filters, '../data/type_filters')
    save_obj(question_text_df, '../data/question_text_df')
    save_obj(answer_text, '../data/answer_text')
    print("Created datasets")

# Dowloading history for prior
try:
    df_history = load_obj('../data/df_history')
except:
    df_history = utils.algo_utils.create_history(traffic_cat, question_text_df)
    save_obj(df_history, '../data/df_history')
    print("Created history")

# ============= GETTING TEACHER DATA ========== #
print("#"*50) #
print('Collecting data from teacher (MaxMI algorithm)... \n')

try:
    data = tl.files.load_npy_to_any(name='../teacher_dagger/_tmp.npy')
    #data = tl.files.load_npy_to_any(name='../teacher_dagger/s500_p2a0.2_p3a0.1_pidk0.1_a1.0_tmp.npy')
    state_list = data['state_list']
    question_list = data['act']
    print('Data found and loaded')
except:
    # Run MaxMI and get the output set of (state, question) for number of products defined in flags (default: 200)
    print("Data not found; teacher is creating it \n")

    # Get question list and state_list of the form : {"q1":[a1,a2], "q2":[a3], "q3":[a4, a6] ..}
    # Use history
    a_hist=1
    state_list, question_list = dagger_utils.get_data_from_teacher(products_cat,
                                                                    traffic_cat,
                                                                    purchased_cat,
                                                                    a_hist,
                                                                    df_history,
                                                                    question_text_df,
                                                                    answer_text,
                                                                    FLAGS.threshold,
                                                                    FLAGS.in_maxMI_size)
    # Save data as _tmp.npy file (to be reused in next iteration)
    if not os.path.exists(os.path.join(os.path.curdir, "../teacher_dagger/")):
        os.makedirs(os.path.join(os.path.curdir, "../teacher_dagger/"))
    print("Saving dagger to {}\n".format(os.path.join(os.path.curdir, "../teacher_dagger/")))
    tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': question_list}, name='_tmp.npy')
    print('Saved teacher data')

# ============= SETTING UP CHECKPOINT DIRECTORY ========== #
# Set up output directory for models and summaries
if FLAGS.run_name is None:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs/h1{}_h2{}_ts{}".format(FLAGS.h1, FLAGS.h2,timestamp)))
else:
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs", FLAGS.run_name))
if not os.path.exists(out_dir):
        os.makedirs(out_dir)
print("Writing to {}\n".format(out_dir))

checkpoint_path = out_dir + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback for later saving of the model
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=50)


# ============= TRAINING WITH INITIAL TEACHER DATA ========== #
print('Converting labels to index encoding..')
one_ind_labels = dagger_utils.get_index_question(question_list, filters_def_dict)
print('Shape of one index labels {}'.format(np.shape(one_ind_labels)))
number_filters = len(filters_def_dict.keys())
print('Number available filters is {}'.format(number_filters))

print('Converting state to one hot encoding..')
onehot_state_list = []
mask_list = []

# Create a mask list to prevent the network from asking the same questions
for state in state_list:
    question_asked = state.keys()
    one_ind_questions_asked = dagger_utils.get_index_question(question_asked, filters_def_dict)
    mask = np.ones(number_filters)
    for q in one_ind_questions_asked:  # If question was already asked, set corresponding mask value to 0
        mask[q] = 0
    # Get one hot state
    onehot_state = dagger_utils.get_onehot_state(state, filters_def_dict)
    onehot_state_list.append(np.asarray(onehot_state))
    mask_list.append(mask)

# Convert to array
one_hot_state_list = np.asarray(onehot_state_list)
mask_list = np.asarray(mask_list)
length_state = np.size(onehot_state_list[0])

print('Length of the one-hot state vector is {}'.format(length_state))
print('Total number of initial (state, question) training pairs is {}'.format(len(onehot_state_list)))

# Model definition
print('Init model')
model = create_model(number_filters, length_state, h1 = FLAGS.h1, h2=FLAGS.h2)

# Print summary of parameters
model.summary()

# Fit the model
model_history = model.fit([one_hot_state_list, mask_list],
                            one_ind_labels,
                            epochs=FLAGS.n_epochs_init,
                            batch_size=FLAGS.batch_size,
                            validation_split=FLAGS.val_split, 
                            verbose=2,
                            callbacks=[cp_callback])

# For the plots
model_history_train_loss = model_history.history['loss']
model_history_val_loss = model_history.history['val_loss']
model_history_train_acc = model_history.history['acc']
model_history_val_acc = model_history.history['val_acc']
model_history_epochs = model_history.epoch

plt.figure(figsize=(16,10))
val = plt.plot(model_history_epochs, model_history_val_loss,'--', label='Validation set'.title())
plt.plot(model_history_epochs, model_history_train_loss, color=val[0].get_color(), label='Training set'.title())
plt.xlabel('Epochs')
plt.ylabel('Loss'.title())
plt.legend()
plt.xlim([0,max(model_history_epochs)])
plt.savefig(checkpoint_dir+"/loss-Init.png", dpi=300)

plt.figure(figsize=(16,10))
val = plt.plot(model_history_epochs, model_history_val_acc,'--', label='Validation set'.title())
plt.plot(model_history_epochs, model_history_train_acc, color=val[0].get_color(), label='Training set'.title())
plt.xlabel('Epochs')
plt.ylabel('Accuracy'.title())
plt.legend()
plt.xlim([0,max(model_history_epochs)])
plt.savefig(checkpoint_dir+"/acc-Init.png", dpi=300)

# plt.show()  # If show the plot, must manually close the window to resume the execution of the program
print(model_history.history.keys())

# ============= COLLECT MORE DATA (EXPLORING NEW STATES) & RETRAIN NETWORK AT EACH EPISODE ========== #
output_file = open(checkpoint_dir+'/results.txt', 'w')
n_episodes = FLAGS.n_episodes

for episode in range(n_episodes):

    # Get latest checkpoint of the network
    # latest = tf.train.latest_checkpoint(checkpoint_dir+'/')  #TODO does not work apparently, try to find solution once start training
    print('Loading the latest model')
    # use latest checkpoint manually
    latest = out_dir+'/cp.ckpt' 
    
    # Restore the model from the checkpoint
    model = create_model(number_filters, length_state, h1=FLAGS.h1, h2=FLAGS.h2)
    model.load_weights(latest)
    
    # Start the imitation learning
    print("#" * 50)
    print("# Episode: %d start" % episode)
    
    # Sample one target product
    y = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size=1)[0]
    
    # Sample the possible answers for this product:
    # Creates a dictionary of the form {'q1': [a1], 'q2': [a2, a3], etc.}
    answers_y = sampler.sample_answers(y, products_cat, p_idk=0.1, p_2a=0.1, p_3a=0.1)
    state = {}  # Initial state

    # Loop until obtain all possible states (until # products in products set < threshold)
    while True: 
        # Get list of questions already asked
        question_asked = state.keys()
        # Convert to one-hot
        one_ind_questions_asked = dagger_utils.get_index_question(question_asked, filters_def_dict)
        # Create the mask before the softmax layer (cannot ask twice the same question)
        mask = np.ones(number_filters)
        for q in one_ind_questions_asked:  # If question was already asked, set corresponding mask value to 0
            mask[q] = 0
        
        # Get one hot state encoding
        onehot_state = dagger_utils.get_onehot_state(state, filters_def_dict)

        # If not first question, add current mask and state to train data
        # We do not save the first question because it is always the same (for optimal with no history into account) 
        if not state == {}:
            # Get the question that the teacher would have asked given current state
            q_true, done = dagger_utils.get_next_question_opt(state,
                                                                products_cat,
                                                                traffic_cat,
                                                                purchased_cat,
                                                                FLAGS.threshold)
            if done is True:  # stop if (# products < threshold) is True
                break
            else: 
                # Reshape to be able to concatenate
                onehot_state = np.reshape(onehot_state, (1, -1))
                mask = np.reshape(mask, (1, -1))
                # Append the new state s(t) to the training_set
                one_hot_state_list = np.concatenate((one_hot_state_list, onehot_state))
                mask_list = np.concatenate((mask_list, mask))
                one_ind_question = dagger_utils.get_index_question([q_true], filters_def_dict)[0]
                one_ind_labels = np.append(one_ind_labels, one_ind_question)
        
        onehot_state = np.reshape(onehot_state, (1, -1))
        mask = np.reshape(mask, (1, -1))
        # Get predicted question from model for current state
        probas = model.predict({'main_input': onehot_state, 'mask_input': mask})[0]  # Predict the one-hot label
        print(probas)
        onehot_prediction = np.argmax(probas)
        q_pred = sorted(filters_def_dict.keys())[onehot_prediction]  # Get the number of predicted next question
        
        # Update (answer) state according to that prediction
        answers_to_pred = answers_y.get(float(q_pred))  # Get answer (from randomly sample product) to chosen question
        state[q_pred] = list(answers_to_pred)
        print(state)

    print("#" * 50)
    output_file.write('Episode: %02d\t Number or questions: %02d\n' % (episode, len(state)))
    
    # Retrain the model with the new data at the end of the episode
    # Only retrain every 500 episodes
    if episode % 500==0:
        # Last state is not relevant for training, since no predicted next state (question)
        model_history = model.fit([one_hot_state_list, mask_list],
                                    one_ind_labels,
                                    epochs=FLAGS.n_epochs,
                                    batch_size=FLAGS.batch_size,
                                    validation_split=FLAGS.val_split,
                                    verbose=2,
                                    callbacks=[cp_callback])
    
        model_history_epochs = np.append(model_history_epochs, model_history.epoch)
        model_history_train_loss = np.append(model_history_train_loss, model_history.history['loss'])
        model_history_val_loss = np.append(model_history_val_loss, model_history.history['val_loss'])
        model_history_train_acc = np.append(model_history_train_acc, model_history.history['acc'])
        model_history_val_acc = np.append(model_history_val_acc, model_history.history['val_acc'])

        plt.clf()
        plt.figure(figsize=(16,10))
        val = plt.plot(model_history_epochs, model_history_val_loss,'--', label='Validation set'.title())
        plt.plot(model_history_epochs, model_history_train_loss, color=val[0].get_color(), label='Training set'.title())
        plt.xlabel('Epochs')
        plt.ylabel('Loss'.title())
        plt.legend()
        plt.xlim([0,max(model_history_epochs)])
        plt.savefig(checkpoint_dir+"/loss-E{}.png".format(episode), dpi=300)

        plt.clf()
        plt.figure(figsize=(16,10))
        val = plt.plot(model_history_epochs, model_history_val_acc,'--', label='Validation set'.title())
        plt.plot(model_history_epochs, model_history_train_acc, color=val[0].get_color(), label='Validation set'.title())
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy'.title())
        plt.legend()
        plt.xlim([0,max(model_history_epochs)])
        plt.savefig(checkpoint_dir+"/acc-E{}.png".format(episode), dpi=300)
        """
        # Plot loss and accuracy
        dagger_utils.plot_history(model_history, key='loss')
        plt.savefig(checkpoint_dir+'/results/'+"loss-E{}.png".format(episode), dpi=900)
        dagger_utils.plot_history(model_history, key='acc')
        plt.savefig(checkpoint_dir+'/results/'+"acc-E{}.png".format(episode), dpi=900)
        """
