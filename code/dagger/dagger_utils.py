import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

import utils.algo_utils as algo_utils
from greedy.MaxMI_Algo import max_info_algorithm, opt_step
import greedy.MaxMI_Algo as MaxMI
from utils.sampler import sample_answers
from utils.algo_utils import get_proba_Y_distribution
from utils.build_answers_utils import question_id_to_text, answer_id_to_text

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
    question_set_new = set(algo_utils.get_questions(product_set)) 
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
    questions = [int(float(q)) for q in sorted(filters_def_dict.keys())]
    onehot_state = []
    for q in questions:
        # Get all sorted possible answers
        # some questions have an answer type object and other a normal array
        if filters_def_dict[str(float(q))].dtype == object:
            all_a = sorted(filters_def_dict[str(float(q))].item())
        else:
            all_a = sorted(filters_def_dict[str(float(q))])
        # if q has been answered in state
        if q in state.keys():
            a = state[q]  #get answers from that question
            if not isinstance(a,list):
                a = a.tolist()
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

def plot_history(epochs, metric_history_val, metric_history_train, x_breaks, title, filename):
    """ Helper for plot history.

    Args:
        epochs: number of epochs
        metric_history_val: list of metric values on the validation set per epoch.
        metric_history_train: list of metric values on the train set per epoch.
        x_breaks: x-coord of the vertical lines in the plot to distinguish the different rounds. 
        title: title of the plot
        filename: filename to save the plot.
    """
    plt.figure(figsize=(16,10))
    plt.plot(np.arange(1, epochs+1), metric_history_val,'--', color='b', label='Validation set'.title())
    plt.plot(np.arange(1, epochs+1), metric_history_train, color='b', label='Training set'.title())
    for xc in x_breaks:
        plt.axvline(x=(xc-1), linestyle='--', color='k', linewidth=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(title.title())
    plt.legend()
    plt.xlim([1, epochs])
    plt.xticks(np.arange(1, epochs+1))
    plt.savefig(filename, dpi=300)

def dagger_get_questions(y, answers_y, model, question_text_df, answer_text_df, filters_def_dict, products_cat, number_filters):
    """ This function returns the list of questions for one sampled user with
    one trained instance of dagger.

    Note:
        You have to first trained the model and initialize it.

    Args:
        y: target productID for the sampled user
        answers_y: sampled answers for this product.
        model: trained model
        question_text_df: table to link questionId to text [PropertyDefinition	PropertyDefinitionId]
        answer_text_df: table to link answerId to text [answer_id	question_id	answer_text]
        filters_def_dict: dict where key is questionId value is array of all possible (modified) answers
        products_cat: extract of product catalog for category 6
        number_filters: number of available questions
    Returns:
        final_question_list: sequence of questionId to ask
        product_set: final product list
        y: product chosen as input of algo
        final_question_text_list:  sequence of questionText to ask
        answer_text_list: answers for each final question
    """
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    # Restore the model from the checkpoint
    # Initial state
    state = {}  
    # Loop until obtain all possible states (until # products in products set < threshold)
    while True: 
        # Get list of questions already asked
        question_asked = state.keys()
        # Convert to one-hot
        one_ind_questions_asked = get_index_question(question_asked, filters_def_dict)
        # Create the mask before the softmax layer (cannot ask twice the same question)
        mask = np.ones(number_filters)
        for q in one_ind_questions_asked:  # If question was already asked, set corresponding mask value to 0
            mask[q] = 0
        # Get one hot state encoding
        onehot_state = get_onehot_state(state, filters_def_dict)
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

def dagger_one_step(model, state, number_filters, filters_def_dict):
    """ Find the next optimal question from a trained model.

    Args:
        model: trained model
        state: input state to predict the next question
        number_filters: number of available questions
        filters_def_dict: dict where key is questionId value is array of all possible (modified) answers
    
    Returns:
        q_pred: id of the next question to ask.
    """
    # Get list of questions already asked
    question_asked = state.keys()
     # Convert to one-hot
    one_ind_questions_asked = get_index_question(question_asked, filters_def_dict)
    # Create the mask before the softmax layer (cannot ask twice the same question)
    mask = np.ones(number_filters)
    for q in one_ind_questions_asked:  # If question was already asked, set corresponding mask value to 0
        mask[q] = 0
    # Get one hot state encoding
    onehot_state = get_onehot_state(state, filters_def_dict)
    onehot_state = np.reshape(onehot_state, (1, -1))
    mask = np.reshape(mask, (1, -1))
    # Get predicted question from model for current state
    probas = model.predict({'main_input': onehot_state, 'mask_input': mask})[0]  # Predict the one-hot label
    onehot_prediction = np.argmax(probas)
    q_pred = sorted(filters_def_dict.keys())[onehot_prediction]  # Get the number of predicted next question
    return q_pred


if __name__=="__main__":
    """ The main procedure of this file is used to launch one run of teacher training.
    It runs the optimal algorithm and saves the results to a file.
    By launching this script several time with different parameters you get several teacher files
    that can later be aggregated in order to create the initial training dataset. 

    See ReadMe for more details.
    """
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
    size = args.size if args.size else 1
    a_hist = args.a_hist if args.a_hist else 0.0
    p_idk = args.pidk if args.pidk else 0.0
    p_2a = args.p2a if args.p2a else 0.0
    p_3a = args.p3a if args.p3a else 0.0
    state_list, question_list = get_data_from_teacher(products_cat,
                                                    traffic_cat, 
                                                    purchased_cat, 
                                                    a_hist,
                                                    df_history,
                                                    question_text_df,
                                                    answer_text,
                                                    threshold,
                                                    size)
    if not os.path.exists(os.path.join(os.path.curdir, "../teacher_dagger/")):
            os.makedirs(os.path.join(os.path.curdir, "../teacher_dagger/"))
    print("Saving dagger to {}\n".format(os.path.join(os.path.curdir, "../teacher_dagger/")))
    timestamp = str(int(time.time()))
    tl.files.save_any_to_npy(save_dict={'state_list': state_list, 'act': question_list},
                             name = '../teacher_dagger/s{}_p2a{}_p3a{}_pidk{}_a{}_{}_tmp.npy'.format(size, p_2a, p_3a, p_idk, a_hist, timestamp))

