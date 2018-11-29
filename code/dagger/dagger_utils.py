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
from greedy.MaxMI_Algo import max_info_algorithm, opt_step
import greedy.MaxMI_Algo as MaxMI
from utils.sampler import sample_answers
from utils.algo_utils import get_proba_Y_distribution

def get_products(state, product_set, traffic_set=[], purchased_set=[]):
    """ Update the data tables from the state dict
    Args:
        state: {question1:answer1, question2: answer2, ...}
        product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
    Returns:
        result_df: updated product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: updated traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: updates purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
    """
    result_df = product_set.copy()
    for q, a in state.items():
        result_df, traffic_set, purchased_set = algo_utils.select_subset(result_df, question = q, answer = a, traffic_set=traffic_set, purchased_set=purchased_set)
    return result_df, traffic_set, purchased_set


def get_next_question_opt(state, product_set, traffic_set, purchased_set, threshold):
    """ Compute the true next question, according to entropy principle, given the history of previous questions and answers.
    Args:
        state: {question1:answer1, question2: answer2, ...}
        product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        threshold: max length of final set of products
    Returns:
        next_question: next optimal question to ask
        done: boolean variable to indicate if reached the minimal set of remaining product
    """
    product_set, traffic_set, purchased_set = get_products(state, product_set,traffic_set, purchased_set)
    n = len(np.unique(product_set["ProductId"]))
    print('remaining prod {}'.format(n))
    question_set_new = set(algo_utils.get_filters_remaining(product_set)) 
    question_set = question_set_new.difference(state.keys()) # state keys is the list of questions asked 
    if n < threshold :
        done = True  # the remain product_set is smaller than threshold
        next_question = 0
    else:
        done = False
        next_question = MaxMI.opt_step(question_set, product_set, traffic_set, purchased_set)
    return next_question, done


def get_data_from_teacher(products_cat, traffic_cat, purchased_cat, a_hist, df_history, question_text_df, answer_text_df, threshold, size=200, p_idk=0.1, p_2a=0.1, p_3a=0.1):
    """ Compute the trajectory for all the products following the entropy principle, and divide them in states and actions.
    Args:
        product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        a_hist (default = 0): parameter to determine the importance of history filters, the higher the more important history is. 0 means no history
        df_history (default = 0): history table obtained with algo_utils.create_history(traffic_cat, question_text_df) [ProductId	text	frequency]
        question_text_df: table to link questionId to text [PropertyDefinition	PropertyDefinitionId]
        answer_text_df: table to link answerId to text [answer_id	question_id	answer_text]
        threshold: max length of final set of products
        size: number of trajectories to produce
        p_idk (default 0.1): additional probability of answering "I dont know" to a given question
        p_2a (default 0.3): probability of giving 2 answers to a given question (true answer and random other one)
        p_3a (default 0.15): probability of giving 3 answers to a given question.
    Returns:
        state_list: questions, answers made by MaxMI algorithm
        question_list: action taken for each state
    """
    # Optimization: compute first_questions outside product loop
    first_questions = []
    first_question_set = set(algo_utils.get_questions(products_cat))
    n_first_q = 3 
    print("Optimization: computing first {} questions without history beforehand".format(n_first_q))
    for i in range(n_first_q):
        first_question = opt_step(first_question_set, products_cat, traffic_cat, purchased_cat, a_hist, df_history)
        first_questions.append(first_question)
        first_question_set = first_question_set.difference(set(first_questions))

    # Select a subset of size products from the y_distribution
    state_list = []
    all_questions_list = []
    p_y = get_proba_Y_distribution(products_cat, purchased_cat, alpha=1)["final_proba"].values
    y_array = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size, p = p_y)

    # For each product compute the trajectory following the MaxMI algorithm
    for y in y_array:
        answers_y = sample_answers(y, products_cat, p_idk, p_2a, p_3a) 
        question_list, _, _, _, _ = max_info_algorithm(products_cat, 
                                                        traffic_cat, 
                                                        purchased_cat,
                                                        question_text_df,
                                                        answer_text_df,
                                                        threshold, 
                                                        y, 
                                                        answers_y,
                                                        a_hist,
                                                        df_history,
                                                        first_questions)
        # Divide the entire trajectory in {state, action}
        history = {}     # first state is state zero
        state_list.append(history.copy())
        for q in question_list[: -1]:
            answers = answers_y.get(q)
            history[q] = answers
            state_list.append(history.copy())
            all_questions_list.append(q)
        all_questions_list.append(question_list[-1])
    return state_list, all_questions_list


def get_onehot_state(state, filters_def_dict):
    """ Compute the one-hot vector state from state:
    Args:
        state: {"q1":[a1,a2], "q2":[a3], ..}
        filters_def_dict:
    Returns:
        onehot_state: one-hot vector state ([0,0,1,1,0,0,...,0,0])
    """
    questions = sorted(filters_def_dict.keys())
    onehot_state = []
    for q in questions:
        # Get all sorted possible answers
        # some questions have an answer type object and other a normal array
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

def get_index_question(question_list, filters_def_dict):
    """ Compute a list of indices to represent the question asked at each time:
    Args:
         question_list: questions considered
         filters_def_dict: dict where key is questionId, value is array of all possible (modified) answers
    Returns:
        all_one_hot: list of indices indicating the question asked at each time
    """
    questions_sorted=np.asarray(sorted(filters_def_dict.keys()))
    all_indices = []
    for q in question_list:
        i = np.where(questions_sorted==str(float(q)))[0][0]
        all_indices.append(i)
    return np.asarray(all_indices)


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
    try:
        df_history = load_obj('../data/df_history')
    except:
        df_history = algo_utils.create_history(traffic_cat, question_text_df)
        save_obj(df_history, '../data/df_history')
        print("Created history")

    
    threshold = 50
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size",
                    help="number of products to teach on", type=int)
    parser.add_argument("-a", "--a_hist",
                    help="alpha parameter, 0 means no history filters used, ow the bigger it is the more importance is given to history", type=float)
    parser.add_argument("-pidk", "--pidk",
                    help="proba of user answering I don't know to a question", type=float)
    parser.add_argument("-p2a", "--p2a",
                    help="proba of user giving 2 answers to a question", type=float)
    parser.add_argument("-p3a", "--p3a",
                    help="proba of user giving 3 answers to a question", type=float)
    args = parser.parse_args()
    size = args.size if args.size else 200
    a_hist = args.a_hist if args.a_hist else 0.0
    p_idk = args.pidk if args.pidk else 0.0
    p_2a = args.p2a if args.p2a else 0.0
    p_3a = args.p3a if args.p3a else 0.0
    """
    state_list, question_list = get_data_from_teacher(products_cat,
                                                    traffic_cat, 
                                                    purchased_cat, 
                                                    a_hist,
                                                    df_history,
                                                    question_text_df,
                                                    answer_text,
                                                    threshold,
                                                    size)"""
    if not os.path.exists(os.path.join(os.path.curdir, "../runs_dagger/")):
            os.makedirs(os.path.join(os.path.curdir, "../runs_dagger/"))
    print("Saving dagger to {}\n".format(os.path.join(os.path.curdir, "../runs_dagger/")))
    tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': question_list}, name = '../runs_dagger/s{}_p2a{}_p3a{}_pidk{}_a{}_tmp.npy'.format(size, p_2a, p_3a, p_idk, a_hist))