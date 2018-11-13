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

# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)



## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("run_name", None, "saving")
tf.flags.DEFINE_integer("threshold", 50, "Length of the final subset of products")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_float("val_split", 0.3, "Fraction used for validation during training")
tf.flags.DEFINE_integer("n_epochs", 20, "Number of epochs")
tf.flags.DEFINE_integer("checkpoint_every", 100, "checkpoint every")
#choose the best training checkpoint as a pretrained net
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523379413/checkpoints/", "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


if __name__=='__main__':
    ###================= Load the product catalogue
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


    ###================= Get demonstration data
    """TO USE IN PRE-TRAINING"""
    print("#"*50)
    print('Collecting data from teacher (MaxMI) \n')

    try:
        data = tl.files.load_npy_to_any(name='old_tmp.npy')
        state_list = data['state_list']
        question_list = data['act']
        print('Data found') # MEL: last instruction otherwise always printed
    except:
        print("Data not found, asking the teacher to create it \n")

        #state = {"q1":[a1,a2], "q2":[a3], ..}
        state_list, question_list = dagger_utils.get_data_from_teacher(products_cat, 
                                                          traffic_cat,
                                                          purchased_cat,
                                                          question_text_df,
                                                          answer_text,
                                                          FLAGS.threshold)
        #for all products run MaxMI and get the set of (state, question) it made

        print('Saving data')
        tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': question_list}, name = '_tmp.npy')

    ###============ Set up checkpoint directory
    # Output directory for models and summaries
    if FLAGS.run_name is None:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs", timestamp))
    else:
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs", FLAGS.run_name))
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.makedirs(out_dir+'/results')
    print("Writing to {}\n".format(out_dir))
    checkpoint_path = out_dir+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create checkpoint callback for later saving of the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=10)
    
    ###===================== Initial training of the model using data from teaching
    question_list = question_list[0] #TODO once we have full data remove this line (was wrongly saved but i corrected it)
    
    print('Converting labels to one hot')
    one_hot_labels = dagger_utils.get_onehot_question(question_list, filters_def_dict)
    print('Shape of one hot labels {}'.format(np.shape(one_hot_labels)))
    number_filters = len(filters_def_dict.keys())
    print('Number available filters is {}'.format(number_filters))

    print('Converting state to one hot')
    onehot_state_list = []
    mask_list = []
    for state in state_list:
        # to prevent the network from asking the same questions
        question_asked = state.keys()
        one_hot_questions_asked = dagger_utils.get_onehot_question(question_asked, filters_def_dict)
        mask = np.ones(number_filters)
        for q in one_hot_questions_asked:
            mask[q] = 0
        # get one hot state
        onehot_state = dagger_utils.get_onehot_state(state, filters_def_dict)
        # add to data
        onehot_state_list.append(np.asarray(onehot_state))
        mask_list.append(mask)
    
    one_hot_state_list = np.asarray(onehot_state_list[:-1])
    mask_list = np.asarray(mask_list[:-1])
    
    length_state = np.size(onehot_state_list[0])
    
    print('Length of the one-hot state vector is {}'.format(length_state))

    print('Total number of initial (state, question) training pairs is {}'.format(len(onehot_state_list)))
    
    # Model definition
    print('Init model')
    model = create_model(number_filters, length_state)
    
    # Print summary of parameters
    model.summary()

    # Fit the model
    model_history = model.fit([one_hot_state_list, mask_list] ,
                                one_hot_labels,
                                epochs=FLAGS.n_epochs,
                                batch_size=FLAGS.batch_size,
                                validation_split=FLAGS.val_split, #or validation_data=(val_state, val_labels),
                                verbose=2,
                                callbacks = [cp_callback])
    # Print first training plots
    dagger_utils.plot_history(model_history)
    plt.savefig(checkpoint_dir+'/results/'+"plot-Init.png", dpi=900)
    #plt.show() # if you show the plot you have to manually close the window to resume the execution of the program.
    

    ###===================== Simulate from model and opt model and retrain at each episode
    output_file = open(checkpoint_dir+'/results/results.txt', 'w')
    n_episode=1
    for episode in range(n_episode):

        #Get latest checkpoint of the net policy
        # TODO does not work i don't know why, try to find solution
        # latest = tf.train.latest_checkpoint(checkpoint_dir+'/') 
        
        # use latest checkpoint manually
        latest = out_dir+'/cp-0020.ckpt' 
        
        # restore the model from the checkpoint
        model = create_model(number_filters, length_state)
        model.load_weights(latest)
        
        # start the game
        print("#" * 50)
        print("# Episode: %d start" % episode)
        
        # sample one target product
        y = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = 1)[0]
        
        # sample the answers for this product
        answers_y = sampler.sample_answers(y, products_cat, p_idk=0.1, p_2a = 0.1, p_3a=0.1) # dict {'q1': [a1], 'q2': [a2, a3]}
        state = {} # initial state

        #loop until end of the episode
        while True: 
            
            # get list of questions already asked
            question_asked = state.keys()
            # to one-hot
            one_hot_questions_asked = dagger_utils.get_onehot_question(question_asked, filters_def_dict)
            # create the mask before the softmax layer (not allowed to ask 2 same question)
            mask = np.ones(number_filters)
            for q in one_hot_questions_asked:
                mask[q] = 0
            
            # get one hot state
            onehot_state = dagger_utils.get_onehot_state(state, filters_def_dict)

            # if not first question add current mask and state to train data
            # not saving first question cause always the same for optimal with {} history
            if not state == {}:
                one_hot_state_list = np.append(one_hot_state_list, onehot_state) # append the state s(t) to the training_set
                mask_list = np.append(mask_list, mask)
            
                # get the question that the teacher would have asked to current state
                # and this question to training data
                q_true, done = dagger_utils.get_next_question_opt(state, products_cat, traffic_cat, purchased_cat, FLAGS.threshold)
                if done is True:
                    break
                else:
                    one_hot_labels = np.append(one_hot_labels, dagger_utils.get_onehot_question([q_true], filters_def_dict)[0])
            
            # get predicted question from model for current state
            probas = model.predict([np.reshape(onehot_state, (1,-1)), np.reshape(mask, (1,-1))])[0] #predict the one-hot label
            print(probas)
            onehot_prediction = np.argmax(probas)
            q_pred = sorted(filters_def_dict.keys())[onehot_prediction] # get the question number
            answers_to_pred = answers_y.get(int(q_pred))
            
            # update state according to that prediction
            state[q_pred] = list(answers_to_pred) # this is the next input state
            print(state)

        print("#" * 50)
        output_file.write('Episode: %02d\t Number or questions: %02d\n' % (episode, len(state)))
        
        # At the end of the episode retrain the model with the new data.
        model_history = model.fit(one_hot_state_list,
                                one_hot_labels,
                                epochs=FLAGS.n_epochs,
                                batch_size=FLAGS.batch_size,
                                validation_split=FLAGS.val_split, #or validation_data=(val_state, val_labels),
                                verbose=2,
                                callbacks = [cp_callback])
        
        # plot the new loss
        dagger_utils.plot_history(model_history)
        plt.savefig(checkpoint_dir+'/results/'+"plot-E{}.png".format(episode), dpi=900) 

