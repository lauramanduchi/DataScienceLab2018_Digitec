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
from greedy.MaxMI_Algo import max_info_algorithm
import greedy.MaxMI_Algo as MaxMI
from utils.sampler import sample_answers


def get_products(state, product_set, traffic_set=[], purchased_set=[]):
    """ from the state dict get the remaining products """
    result_df = product_set.copy()
    for q, a in state.items():
        result_df, traffic_set, purchased_set = algo_utils.select_subset(result_df, question = q, answer = a, traffic_set=traffic_set, purchased_set=purchased_set)
    return result_df, traffic_set, purchased_set


def get_next_question_opt(state, product_set, traffic_set, purchased_set, threshold):
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
        next_question = MaxMI.opt_step(question_set, product_set, traffic_set, purchased_set)
    return next_question, done


def get_data_from_teacher(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text, threshold, size=200, p_idk=0.1, p_2a = 0.1, p_3a=0.1): 
    """ Compute the trajectory for all the products following the entropy principle, and divide them in states and actions.
    Args:
        original product catalog, traffic table and purchased articles from the selected category.
    Returns:
        state_list (questions, answers made) and question_list (actions)
    """
    state_list = []
    all_questions_list = []
    for y in np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size):
        answers_y = sample_answers(y, products_cat, p_idk, p_2a, p_3a) 
        question_list, _, _, _, _ = max_info_algorithm(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text,
                            threshold, y,  answers_y)
        # first state in state zero
        history = {}
        state_list.append(history)
        for q in question_list[: -1]:
            answers = answers_y.get(q)
            history[q] = answers
            state_list.append(history)
            all_questions_list.append(q)
        all_questions_list.append(question_list[-1])
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
        i = np.where(questions_sorted==str(q))[0][0]
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


if __name__=="__main__":
    import tensorlayer as tl
    from utils.load_utils import load_obj, save_obj
    import argparse
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
        print("Data not found. Create datasets first please")
    
    threshold = 50
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size",
                    help="number of products to teach on", type=int)
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
    size_test = args.size if args.size else 200
    use_history = args.use_history if args.use_history else False
    alpha = args.alpha if args.alpha else 0.0
    p_idk = args.pidk if args.pidk else 0.0
    p_2a = args.p2a if args.p2a else 0.0
    p_3a = args.p3a if args.p3a else 0.0
    
    state_list, question_list = get_data_from_teacher(products_cat, \
                                                    traffic_cat, \
                                                    purchased_cat, \
                                                    question_text_df, \
                                                    answer_text, \
                                                    threshold, \
                                                    size)
    
    tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': question_list}, name = 's{}_p2a{}_p3a{}_pidk{}_a{}_tmp.npy'.format(size, p_2a, p_3a, p_idk, alpha))