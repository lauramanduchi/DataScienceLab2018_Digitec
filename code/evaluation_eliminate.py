import time
import os
import numpy as np
from load_utils import *
from eliminate import max_eliminate_algorithm
from RandomBaseline import random_baseline, get_distinct_products
import pandas as pd
from init_dataframes import init_df
import algo_utils
import argparse
from sampler import sample_answers
import warnings

# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)

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

time.time()
t = time.strftime('%d%b%y_%H%M%S')
print("Started on: {}".format(time.strftime('%d-%b-%y at %H:%M:%S')))
cwd = os.getcwd()
checkpoint_dir = cwd+'/../runs_eliminate/' + t + '/'
os.makedirs(checkpoint_dir, 0o777)
print('Saving to ' + checkpoint_dir)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size",
                    help="number of products to test on", type=int)
parser.add_argument("-pidk", "--pidk",
                    help="proba of user answering I don't know to a question", type=float)
parser.add_argument("-p2a", "--p2a",
                    help="proba of user giving 2 answers to a question", type=float)
parser.add_argument("-p3a", "--p3a",
                    help="proba of user giving 2 answers to a question", type=float)

args = parser.parse_args()
size_test = args.size if args.size else 25    
p_idk = args.pidk if args.pidk else 0.0
p_2a = args.p2a if args.p2a else 0.0
p_3a = args.p3a if args.p3a else 0.0

y_array = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size_test)
threshold = 50
length_opt = []
length_rdm = []
opt_quest = []
rdm_quest = []
opt_quest_text = []
rdm_quest_text = []
opt_answer_text_list = []
rdm_answer_text_list = []
with open(checkpoint_dir +'/lengths.csv', 'w+') as f:
    f.write("opt, random \n")
with open(checkpoint_dir +'/quest.csv', 'w+') as f:
    f.write("opt, random \n")
for y in y_array:
    answers_y = sample_answers(y, products_cat, p_idk = p_idk, p_2a = p_2a, p_3a = p_3a)
    final_question_list, product_set, y, final_question_text_list, answer_text_list = max_eliminate_algorithm(products_cat, traffic_cat, purchased_cat,
                                                                                                              question_text_df, answer_text_df,
                                                                                                              threshold, y, answers_y)
    print('the length of optimal eliminate filter was {}'.format(len(final_question_list)))
    length_opt.append(len(final_question_list))
    opt_quest.append(final_question_list)
    opt_quest_text.append(final_question_text_list)
    opt_answer_text_list.append(answer_text_list)
    rb_final_question_list, rb_product_set, rb_y, \
    rb_final_question_text_list, rb_answer_text_list = random_baseline(products_cat, traffic_cat, purchased_cat,
                                                                 question_text_df, answer_text_df, threshold, y)
    length_rdm.append(len(rb_final_question_list))
    rdm_quest.append(rb_final_question_list)
    rdm_quest_text.append(rb_final_question_text_list)
    rdm_answer_text_list.append(rb_answer_text_list)
    print('the length of random filter was {}'.format(len(final_question_list)))
    rdm_quest.append(final_question_list)
    with open(checkpoint_dir +'/lengths.csv', 'a+') as f:
        f.write('{}, {} \n'.format(length_opt[-1], length_rdm[-1]))
    with open(checkpoint_dir +'/opt_quest.csv', 'a+') as f:
        f.write('{}, {} \n'.format(opt_quest[-1], rdm_quest[-1]))
    with open(checkpoint_dir +'/opt_quest_text.csv', 'a+') as f:
         f.write('{}, {} \n'.format(opt_quest_text[-1], rdm_quest_text[-1]))
    with open(checkpoint_dir +'/opt_answer_text.csv', 'a+') as f:
         #f.write('{}, {} \n'.format(opt_answer_text_list[-1], rdm_answer_text_list[-1]))
         f.write('{} \n'.format(opt_answer_text_list[-1]))
    with open(checkpoint_dir +'/rdm_quest.csv', 'a+') as f:
        f.write('{}, {} \n'.format(rdm_quest[-1], rdm_quest[-1]))
    with open(checkpoint_dir +'/rdm_quest_text.csv', 'a+') as f:
         f.write('{}, {} \n'.format(rdm_quest_text[-1], rdm_quest_text[-1]))
    with open(checkpoint_dir +'/rdm_answer_text.csv', 'a+') as f:
         f.write('{} \n'.format(rdm_answer_text_list[-1]))
with open(checkpoint_dir +'/summary.txt', 'w+') as f:
    f.write('Test set size: {} \n Probability of answering I dont know: {} \n Probability of giving 2 answers: {} Probability of giving 3 answers: {} \n'.format(size_test, p_idk, p_2a, p_3a))
    f.write('Avg number of questions for optimal {} \n'.format(np.mean(np.asarray(length_opt))))
    f.write('Std number of questions for optimal {} \n'.format(np.std(np.asarray(length_opt))))
    f.write('Max number of questions for optimal {} \n'.format(np.max(np.asarray(length_opt))))
    f.write('Min number of questions for optimal {} \n'.format(np.min(np.asarray(length_opt))))
    f.write('\n')
    f.write('Avg number of questions for random {} \n'.format(np.mean(np.asarray(length_rdm))))
    f.write('Std number of questions for random {} \n'.format(np.std(np.asarray(length_rdm))))
    f.write('Max number of questions for random {} \n'.format(np.max(np.asarray(length_rdm))))
    f.write('Min number of questions for random {} \n'.format(np.min(np.asarray(length_rdm))))