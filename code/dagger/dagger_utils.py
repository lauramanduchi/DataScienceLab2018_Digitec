import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import utils.algo_utils as algo_utils
from utils.init_dataframes import init_df
from utils.load_utils import load_obj, save_obj
from greedy.eliminate import max_eliminate_algorithm
import greedy.eliminate as eliminate
from utils.sampler import sample_answers
import warnings

def get_products(state, product_set): #MEL: DONE
    """ from the state dict get the remaining products """
    result_df = product_set.copy()
    for q, a in state.items():
        result_df, _, _ = algo_utils.select_subset(product_set, question = q, answer = a)
    return result_df


def get_next_question_opt(state, product_set, traffic_set, purchased_set, threshold): #TODO CHECK
    """ Compute the true next question, according to entropy principle, given the history of previous questions and answers.
    Args:
        state: {question1:answer1, question2: answer2, ...}
    Returns:
        next_question and boolean variable done
    """
    product_set, traffic_set, purchased_set = get_products(state, product_set)
    question_set_new = set(algo_utils.get_filters_remaining(product_set)) 
    question_set = question_set_new.difference(state.keys()) # state keys is the list of questions asked 
    if len(product_set) < threshold :
        done = True # the remain product_set is smaller than threshold
        next_question = 0
    done = False
    next_question = eliminate.opt_step(question_set, product_set, traffic_set, purchased_set)
    return next_question, done


def get_data_from_teacher(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text, threshold): 
    #MEL CHECKED but not run until the end.
    """ Compute the trajectory for all the products following the entropy principle, and divide them in states and actions.
    Args:
        original product catalog, traffic table and purchased articles from the selected category.
    Returns:
        state_list (questions, answers made) and question_list (actions)
    """
    all_products = products_cat["ProductId"].drop_duplicates().values #MEL: changed drop_duplicates cause several lines per product
    state_list = []
    for y in all_products:
        answers_y = sample_answers(y, products_cat)
        question_list, _, _, _, _ = max_eliminate_algorithm(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text,
                            threshold, y,  answers_y) # question list is the full trajectory of chosen action until end of game
        # first state in state zero
        history = {}
        state_list.append(history)
        for q in question_list:
            answers = answers_y.get(q)
            history[q] = answers
            state_list.append(history)
            print(state_list) #MEL: just for testing purposes
            print(len(get_products(state_list[-1], products_cat))) #MEL: just for testing purposes
    return state_list, question_list


def get_onehot_state(state, filters_def_dict):
    """ Compute the one-hot vector state from state.
    Args:
        state: {"q1":[a1,a2], "q2":[a3], ..}
    Returns:
        one-hot vector state ([0,0,1,1,0,0,...,0,0])
    """
    questions = sorted(filters_def_dict.keys())
    onehot_state = []
    for q in questions:
        print(q)
        #get all sorted possible answers
        #some questions have an answer type object and other a normal array
        if filters_def_dict[q].dtype == object:
            all_a = sorted(filters_def_dict[q].item())
        else:
            all_a = sorted(filters_def_dict[q])
        # if q has been answered in state
        if q in state.keys():
            a = state[q]  #get answers from that question
            if not isinstance(a,list):
                a = [a]
            for a_h in all_a: #for all possible answers of q
                if a_h in a:
                    onehot_state.append(1)
                else:
                    onehot_state.append(0)
        # if q has NOT been answered in state
        else:
            [onehot_state.append(0) for i in range(len(all_a))]
    return onehot_state