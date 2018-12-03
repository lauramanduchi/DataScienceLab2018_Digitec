import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import os
import numpy as np
import pandas as pd
import argparse
import warnings

import dagger.dagger_utils as dagger_utils
from dagger.model import create_model
from utils.algo_utils import get_proba_Y_distribution
from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from dagger.dagger_utils import get_products
from greedy.RandomBaseline import random_baseline
from utils.build_answers_utils import question_id_to_text, answer_id_to_text

""" This module runs the evaluation of Dagger.
For each target product in the test set.
    * Sample the answers
    * Find the optimal list of questions with Dagger
    * Find the list of questions with randomBaseline
    * Save the asked questions
    * Output summary statistics of evaluation

The results file can be found in the runs_dagger/run_number folder.
If run_number is not specified the timestep is used instead.
"""

# ============= GENERAL SETUP =========== #
time.time()
t = time.strftime('%H%M%S')
print("Started on: {}".format(time.strftime('%d-%b-%y at %H:%M:%S')))
cwd = os.getcwd()


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size",
                    help="number of products to test on", type=int)
parser.add_argument("-pidk", "--pidk",
                    help="proba of user answering I don't know to a question", type=float)
parser.add_argument("-p2a", "--p2a",
                    help="proba of user giving 2 answers to a question", type=float)
parser.add_argument("-p3a", "--p3a",
                    help="proba of user giving 3 answers to a question", type=float)
parser.add_argument("-r", "--run",
                    help="checkpoint subfolder name", type=float)

args = parser.parse_args()
size_test = args.size if args.size else 25
p_idk = args.pidk if args.pidk else 0.0
p_2a = args.p2a if args.p2a else 0.0
p_3a = args.p3a if args.p3a else 0.0
run = args.run if args.run else 'default'

suffix = 'pidk{}_p2a{}_p3a{}_s{}_t{}'.format(p_idk,p_2a,p_3a,size_test,t)
checkpoint_dir = cwd+'/../runs_eval_dagger/'  + suffix + '/'
os.makedirs(checkpoint_dir, 0o777)
print('Saving to ' + checkpoint_dir)


# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============== LOAD DATA ============= #
try:
    products_cat = load_obj('../data/products_table')
    traffic_cat = load_obj('../data/traffic_table')
    purchased_cat = load_obj('../data/purchased_table')
    question_text_df = load_obj('../data/question_text_df')
    answer_text_df = load_obj('../data/answer_text')
    filters_def_dict = load_obj('../data/filters_def_dict')
    print("Loaded datasets")
except:
    print("Datasets not found...")


def dagger_get_questions(y, answers_y, model, filters_def_dict, products_cat):
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    # Restore the model from the checkpoint
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
        onehot_state = np.reshape(onehot_state, (1, -1))
        mask = np.reshape(mask, (1, -1))
        # Get predicted question from model for current state
        probas = model.predict({'main_input': onehot_state, 'mask_input': mask})[0]  # Predict the one-hot label
        onehot_prediction = np.argmax(probas)
        q_pred = sorted(filters_def_dict.keys())[onehot_prediction]  # Get the number of predicted next question
        question_text = question_id_to_text(q_pred, question_text_df)
        final_question_list.append(int(float(q_pred)))
        final_question_text_list.append(question_text)
        print("DAGGER: Question is: {}".format(question_text))
        # Update (answer) state according to that prediction
        answers_to_pred = answers_y.get(float(q_pred))  # Get answer (from randomly sample product) to chosen question
        answer_text = answer_id_to_text(answers_to_pred, q_pred, answer_text_df)
        print("DAGGER: Answer given was: id:{} text: {}".format(answers_to_pred, answer_text))
        answer_text_list.append(answer_text)
        state[q_pred] = list(answers_to_pred)
        product_set, _, _ = get_products(state, products_cat,[], [])
        if len(np.unique(product_set['ProductId']))<50:
            break
    print('DAGGER: Return {} products.'.format(len(np.unique(product_set['ProductId']))))
    return final_question_list, product_set, y, final_question_text_list, answer_text_list


# ============= MAIN EVALUATION LOOP =============== #
out_dir = '../runs/{}/'.format(run) #h1256_h2128_ts1543838439
checkpoint_model = out_dir+'/cp.ckpt'
print('Loading the latest model from {}'.format(checkpoint_model))
length_state = len(dagger_utils.get_onehot_state({}, filters_def_dict))
number_filters = len(filters_def_dict.keys())
model = create_model(number_filters, length_state, h1=128, h2=122)
model.load_weights(checkpoint_model)

with open(checkpoint_dir +'/parameters.txt', 'w+') as f:
    f.write('Test set size: {} \n Probability of answering I dont know: {} \n Probability of giving 2 answers: {} Probability of giving 3 answers: {} \n'.format(size_test, p_idk, p_2a, p_3a))
    f.write('Loading the latest model from {}'.format(checkpoint_model))
#probabilities of products given traffic
p_y = get_proba_Y_distribution(products_cat, purchased_cat, alpha=1)["final_proba"].values
y_array = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size_test, p = p_y)
threshold = 50
length_opt = []
length_rdm = []
opt_quest = []
rdm_quest = []
opt_quest_text = []
rdm_quest_text = []
opt_answer_text_list = []
rdm_answer_text_list = []
opt_prod_nb = []
rdm_prod_nb = []


with open(checkpoint_dir +'/lengths.csv', 'w+') as f:
    f.write("Dagger, random \n")
with open(checkpoint_dir +'/rdm.csv', 'w+') as f:
    f.write("QuestionId, QuestionText, AnswerText \n")
with open(checkpoint_dir +'/dagger.csv', 'w+') as f:
    f.write("QuestionId, QuestionText, AnswerText \n")

for y in y_array:
    print(y)
    # Sample answers
    answers_y = sample_answers(y, products_cat, p_idk = p_idk, p_2a = p_2a, p_3a = p_3a)
    # Compute optimal list with Dagger
    final_question_list, product_set, y, final_question_text_list, answer_text_list = dagger_get_questions(y,
                                                                                                    answers_y,
                                                                                                    model, 
                                                                                                    filters_def_dict,
                                                                                             products_cat)
    # Save results for Dagger
    print('the length of optimal eliminate filter was {}'.format(len(final_question_list)))
    opt_prod_nb.append(len(np.unique(product_set["ProductId"])))
    length_opt.append(len(final_question_list))
    opt_quest.append(final_question_list)
    opt_quest_text.append(final_question_text_list)
    opt_answer_text_list.append(answer_text_list)

    # Compute random baseline
    rb_final_question_list, rb_product_set, rb_y, rb_final_question_text_list, rb_answer_text_list = random_baseline(products_cat, traffic_cat, purchased_cat,
                                                                 question_text_df, answer_text_df, threshold, y, answers_y)
    rdm_prod_nb.append(len(np.unique(rb_product_set["ProductId"])))
    # Save results for random baseline
    length_rdm.append(len(rb_final_question_list))
    rdm_quest.append(rb_final_question_list)
    rdm_quest_text.append(rb_final_question_text_list)
    rdm_answer_text_list.append(rb_answer_text_list)
    print('the length of random filter was {}'.format(len(rb_final_question_list)))
    rdm_quest.append(rb_final_question_list)
    
    # Save result to external files
    with open(checkpoint_dir +'/lengths.csv', 'a+') as f:
        f.write('{}, {} \n'.format(length_opt[-1], length_rdm[-1]))
    with open(checkpoint_dir +'/dagger.csv', 'a+') as f:
         f.write('{}, {}, {} \n'.format(opt_quest[-1], opt_quest_text[-1], opt_answer_text_list[-1]))
    with open(checkpoint_dir +'/rdm.csv', 'a+') as f:
         f.write('{}, {}, {} \n'.format(rdm_quest[-1], rdm_quest_text[-1],  rdm_answer_text_list[-1]))

# Save the summary statistics of the run in summary.txt
with open(checkpoint_dir +'/summary.txt', 'w+') as f:
    f.write('Test set size: {} \n Probability of answering I dont know: {} \n Probability of giving 2 answers: {} Probability of giving 3 answers: {} \n'.format(size_test, p_idk, p_2a, p_3a))
    f.write('Avg number of questions for dagger {} \n'.format(np.mean(np.asarray(length_opt))))
    f.write('Median number of questions for dagger {} \n'.format(np.median(np.asarray(length_opt))))
    f.write('Std number of questions for dagger {} \n'.format(np.std(np.asarray(length_opt))))
    f.write('Max number of questions for dagger {} \n'.format(np.max(np.asarray(length_opt))))
    f.write('Min number of questions for dagger {} \n'.format(np.min(np.asarray(length_opt))))
    f.write('Max nb of products still left for dagger {} \n'.format(np.max(opt_prod_nb)))
    f.write('Median nb of products still left for dagger {} \n'.format(np.median(opt_prod_nb)))
    f.write('Min nb of products still left for dagger {} \n'.format(np.min(opt_prod_nb)))
    f.write('\n')
    f.write('Avg number of questions for random {} \n'.format(np.mean(np.asarray(length_rdm))))
    f.write('Median number of questions for random {} \n'.format(np.median(np.asarray(length_rdm))))
    f.write('Std number of questions for random {} \n'.format(np.std(np.asarray(length_rdm))))
    f.write('Max number of questions for random {} \n'.format(np.max(np.asarray(length_rdm))))
    f.write('Min number of questions for random {} \n'.format(np.min(np.asarray(length_rdm))))
    f.write('Max nb of products still left for random {} \n'.format(np.max(rdm_prod_nb)))
    f.write('Median nb of products still left for random {} \n'.format(np.median(rdm_prod_nb)))
    f.write('Min nb of products still left for random {} \n'.format(np.min(rdm_prod_nb)))