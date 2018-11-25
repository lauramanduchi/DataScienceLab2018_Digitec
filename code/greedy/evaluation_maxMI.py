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

from utils.algo_utils import get_proba_Y_distribution
from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from greedy.MaxMI_Algo import max_info_algorithm, opt_step
from greedy.RandomBaseline import random_baseline

""" This module runs the evaluation of MaxMI.
For each target product in the test set.
    * Sample the answers
    * Find the optimal list of questions with MaxMI
    * Find the list of questions with randomBaseline
    * Save the asked questions
    * Output summary statistics of evaluation

The results file can be found in the runs_maxMI/run_number folder.
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
parser.add_argument("-hist", "--use_history",
                    help="Boolean to indicate whether to use history data", type=bool)
parser.add_argument("-a", "--alpha",
                    help="alpha parameter, the bigger it is the more importance is given to history", type=float)
parser.add_argument("-pidk", "--pidk",
                    help="proba of user answering I don't know to a question", type=float)
parser.add_argument("-p2a", "--p2a",
                    help="proba of user giving 2 answers to a question", type=float)
parser.add_argument("-p3a", "--p3a",
                    help="proba of user giving 3 answers to a question", type=float)

args = parser.parse_args()
size_test = args.size if args.size else 25
use_history = args.use_history if args.use_history else False
alpha = args.alpha if args.alpha else 0.0
p_idk = args.pidk if args.pidk else 0.0
p_2a = args.p2a if args.p2a else 0.0
p_3a = args.p3a if args.p3a else 0.0


suffix = 'pidk{}_p2a{}_p3a{}_hist{}_s{}_t{}'.format(p_idk,p_2a,p_3a,alpha,size_test,t)
checkpoint_dir = cwd+'/../runs_MaxMI/'  + suffix + '/'
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
    print("Loaded datasets")
except:
    print("Creating datasets...")
    products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text = init_df()
    save_obj(products_cat, '../data/products_table')
    save_obj(traffic_cat, '../data/traffic_table')
    save_obj(purchased_cat, '../data/purchased_table')
    save_obj(filters_def_dict, '../data/filters_def_dict')
    save_obj(type_filters, '../data/type_filters')
    save_obj(question_text_df, '../data/question_text_df')
    save_obj(answer_text, '../data/answer_text')
    print("Created datasets")

df_history = 0
if use_history:
    try:
        df_history = load_obj('../data/df_history')
    except:
        df_history = algo_utils.create_history(traffic_cat, question_text_df)
        save_obj(df_history, '../data/df_history')
        print("Created history")


# ============= INIT VARIABLES ========== #
with open(checkpoint_dir +'/parameters.txt', 'w+') as f:
    f.write('Test set size: {} \n Probability of answering I dont know: {} \n Probability of giving 2 answers: {} Probability of giving 3 answers: {} \n'.format(size_test, p_idk, p_2a, p_3a))
    f.write('History: {} \nAlpha parameter: {}'.format(use_history, alpha))

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
    f.write("MaxMI, random \n")
with open(checkpoint_dir +'/rdm.csv', 'w+') as f:
    f.write("QuestionId, QuestionText, AnswerText \n")
with open(checkpoint_dir +'/optimal.csv', 'w+') as f:
    f.write("QuestionId, QuestionText, AnswerText \n")


# Optimization: compute first_questions outside product loop
first_questions = []
first_question_set = set(algo_utils.get_questions(products_cat))
n_first_q = 3 
print("Optimization: computing first {} questions without history beforehand".format(n_first_q))
for i in range(n_first_q):
    first_question = opt_step(first_question_set, products_cat, traffic_cat, purchased_cat, use_history, df_history, alpha)
    first_questions.append(first_question)
    first_question_set = first_question_set.difference(set(first_questions))


# ============= MAIN EVALUATION LOOP =============== #
for y in y_array:
    print(y)
    # Sample answers
    answers_y = sample_answers(y, products_cat, p_idk = p_idk, p_2a = p_2a, p_3a = p_3a)
    # Compute optimal list with MaxMI
    final_question_list, product_set, y, final_question_text_list, answer_text_list = max_info_algorithm(products_cat, 
                                                                                                         traffic_cat, 
                                                                                                         purchased_cat,
                                                                                                         question_text_df,
                                                                                                         answer_text_df,
                                                                                                         threshold, 
                                                                                                         y, 
                                                                                                         answers_y,
                                                                                                         use_history,
                                                                                                         df_history,
                                                                                                         alpha,
                                                                                                         first_questions)
    # Save results for MaxMI
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
    with open(checkpoint_dir +'/optimal.csv', 'a+') as f:
         f.write('{}, {}, {} \n'.format(opt_quest[-1], opt_quest_text[-1], opt_answer_text_list[-1]))
    with open(checkpoint_dir +'/rdm.csv', 'a+') as f:
         f.write('{}, {}, {} \n'.format(rdm_quest[-1], rdm_quest_text[-1],  rdm_answer_text_list[-1]))

# Save the summary statistics of the run in summary.txt
with open(checkpoint_dir +'/summary.txt', 'w+') as f:
    f.write('Test set size: {} \n Probability of answering I dont know: {} \n Probability of giving 2 answers: {} Probability of giving 3 answers: {} \n'.format(size_test, p_idk, p_2a, p_3a))
    f.write('History: {} \nAlpha parameter: {}'.format(use_history, alpha))
    f.write('Avg number of questions for optimal {} \n'.format(np.mean(np.asarray(length_opt))))
    f.write('Median number of questions for optimal {} \n'.format(np.median(np.asarray(length_opt))))
    f.write('Std number of questions for optimal {} \n'.format(np.std(np.asarray(length_opt))))
    f.write('Max number of questions for optimal {} \n'.format(np.max(np.asarray(length_opt))))
    f.write('Min number of questions for optimal {} \n'.format(np.min(np.asarray(length_opt))))
    f.write('Max nb of products still left for optimal {} \n'.format(np.max(opt_prod_nb)))
    f.write('Median nb of products still left for optimal {} \n'.format(np.median(opt_prod_nb)))
    f.write('Min nb of products still left for optimal {} \n'.format(np.min(opt_prod_nb)))
    f.write('\n')
    f.write('Avg number of questions for random {} \n'.format(np.mean(np.asarray(length_rdm))))
    f.write('Median number of questions for random {} \n'.format(np.median(np.asarray(length_rdm))))
    f.write('Std number of questions for random {} \n'.format(np.std(np.asarray(length_rdm))))
    f.write('Max number of questions for random {} \n'.format(np.max(np.asarray(length_rdm))))
    f.write('Min number of questions for random {} \n'.format(np.min(np.asarray(length_rdm))))
    f.write('Max nb of products still left for random {} \n'.format(np.max(rdm_prod_nb)))
    f.write('Median nb of products still left for random {} \n'.format(np.median(rdm_prod_nb)))
    f.write('Min nb of products still left for random {} \n'.format(np.min(rdm_prod_nb)))