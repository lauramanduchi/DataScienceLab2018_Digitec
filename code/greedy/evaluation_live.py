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

from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from greedy.MaxMI_Algo import max_info_algorithm
from greedy.RandomBaseline import random_baseline
import dagger.dagger_utils as dagger_utils
import utils.build_answers_utils as build_answers_utils
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


threshold = 50
products_set = products_cat.copy()
state = {} # initial state   
done=True 

def get_next_q_user(state, products_cat, traffic_cat, purchased_cat, threshold):
    q, done = dagger_utils.get_next_question_opt(state, products_cat, traffic_cat, purchased_cat, threshold)
    if not done:
        print('id question is {}'.format(q))
        print('possible answers are')
        possible_answers = products_cat.loc[products_cat["PropertyDefinitionId"]==int(q), "answer"] \
                                    .drop_duplicates().values.astype(float)
        print(possible_answers)
        answers = input('choose one answer:')
        try:
            state[q] = [float(x) for x in answers.split()]
        except:
            state[q] = ['idk']
    return done, state

## first question separate
q, done = 347, False
print('id question is {}'.format(q))
print('possible answers are')
possible_answers = products_cat.loc[products_cat["PropertyDefinitionId"]==int(q), "answer"] \
                                    .drop_duplicates().values.astype(float)
text_answers = build_answers_utils.answer_id_to_text(possible_answers, q, answer_text_df)
print(text_answers)
print(possible_answers)
answers = input('choose one answer ID:') # no need TODO because will be done in interface direclty
list = [str(x) for x in answers.split()]
print(list)
try:
    state[q] = [float(x) for x in answers.split()]
except:
    state[q] = ['idk']

done = False

while(not done):
    done, state = get_next_q_user(state, products_cat, traffic_cat, purchased_cat, threshold)
    if done is True:
        print('done')
        products_set = dagger_utils.get_products(state, products_cat)