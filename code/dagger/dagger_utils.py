import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt 
import warnings

import utils.algo_utils as algo_utils
from utils.init_dataframes import init_df
from utils.load_utils import load_obj, save_obj
from greedy.eliminate import max_eliminate_algorithm
from greedy.MaxMI_Algo import max_info_algorithm
import greedy.MaxMI_Algo as MaxMI
import greedy.eliminate as eliminate
from utils.sampler import sample_answers


def get_products(state, product_set, traffic_set=[], purchased_set=[]): #MEL: DONE
    """ from the state dict get the remaining products """
    result_df = product_set.copy()
    for q, a in state.items():
        result_df, traffic_set, purchased_set = algo_utils.select_subset(result_df, question = q, answer = a, traffic_set=traffic_set, purchased_set=purchased_set)
    return result_df, traffic_set, purchased_set


def get_next_question_opt(state, product_set, traffic_set, purchased_set, threshold): #TODO CHECK
    """ Compute the true next question, according to entropy principle, given the history of previous questions and answers.
    Args:
        state: {question1:answer1, question2: answer2, ...}
    Returns:
        next_question and boolean variable done
    """
    product_set, traffic_set, purchased_set = get_products(state, product_set,traffic_set, purchased_set)
    n = len(algo_utils.get_distinct_products(product_set))
    print('remaining prod {}'.format(n))
    question_set_new = set(algo_utils.get_filters_remaining(product_set)) 
    question_set = question_set_new.difference(state.keys()) # state keys is the list of questions asked 
    if n < threshold :
        done = True # the remain product_set is smaller than threshold
        next_question = 0
    else:
        done = False
        #next_question = eliminate.opt_step(question_set, product_set, traffic_set, purchased_set)
        next_question = MaxMI.opt_step(question_set, product_set, traffic_set, purchased_set)
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
    all_questions_list = []
    #for y in all_products: # TODO just to debug remove the other line !!!!!
    for y in np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = 50):
        answers_y = sampler.sample_answers(y, products_cat, p_idk=0.1, p_2a = 0.1, p_3a=0.1) 
        question_list, _, _, _, _ = max_info_algorithm(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text,
                            threshold, y,  answers_y)
        #question_list, _, _, _, _ = max_eliminate_algorithm(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text,
        #                   threshold, y,  answers_y) # question list is the full trajectory of chosen action until end of game
        # first state in state zero
        history = {}
        state_list.append(history)
        for q in question_list:
            answers = answers_y.get(q)
            history[q] = answers
            state_list.append(history)
            all_questions_list.append(q)
        #print(state_list) #MEL: just for testing purposes
        print(len(get_products(state_list[-1], products_cat))[0]) #MEL: just for testing purposes
    return state_list, all_questions_list


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
        #print(q)
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

def get_onehot_question(question_list, filters_def_dict):
    questions_sorted=np.asarray(sorted(filters_def_dict.keys()))
    all_one_hot = []
    for q in question_list:
        #q_one_hot = np.zeros(len(questions_sorted))
        i = np.where(questions_sorted==str(q))[0][0]
        #print(i)
        #q_one_hot[i] = 1
        all_one_hot.append(i)
    return np.asarray(all_one_hot)


# taken from https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
def plot_history(history, name='model', key='loss'):
    plt.figure(figsize=(16,10))
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])