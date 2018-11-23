import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import random
import numpy as np
import pandas as pd
import parmap
import time
import warnings
import multiprocessing

from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.build_answers_utils import question_id_to_text, answer_id_to_text
from utils.sampler import sample_answers

# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)
# about parmap 
#y = [myfunction(x, argument1, mykeyword=argument2) for x in mylist]
# In parallel:
#y = parmap.map(myfunction, mylist, argument1, mykeyword=argument2)


def conditional_entropy(answer, question, product_set, traffic_set, purchased_set):
    """Compute conditional entropy for one particular answer of a question:
        Args:
            answer: given answer
            question: given question
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
            traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
            purchased_set: purchased table [	ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        Returns:
            cond_entropy_y: conditional entropy of the answer
         """
    product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=answer,
                                                            product_set=product_set, traffic_set =traffic_set,
                                                            purchased_set = purchased_set)
    product_ids = product_set["ProductId"].drop_duplicates().values
    try:
        p_product_given_a = algo_utils.get_proba_Y_distribution(product_set, purchased_set, alpha=1)["final_proba"]
    except ZeroDivisionError:
        print('pbm only {} product left'.format(product_ids))
        print(answer)
        print(question)
    prob_y_given_a = [p_product_given_a.loc[product] for product in product_ids]
    cond_entropy_y = np.sum(prob_y_given_a*np.log(prob_y_given_a))

    return cond_entropy_y


def mutual_inf(question, product_set, traffic_set, purchased_set):
    """Compute mutual information for a given question:
        Args:
            question: question considered for computing mutual info
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
            traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
            purchased_set: purchased table [	ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        Returns:
            short_mutual_info: mutual info for that given question
         """
    "speed-up: from 1.5 to 1.3 for each call of mutual inf"

    proba_A = algo_utils.get_proba_A_distribution_none(question, product_set, traffic_set, alpha=1)["final_proba"]
    possible_answers = proba_A.index
    short_mutual_info = proba_A.loc[possible_answers]
    conditional = list(map(lambda x: conditional_entropy(np.asarray([x]), question, product_set, traffic_set, purchased_set), possible_answers))
    short_mutual_info = sum(short_mutual_info * conditional)
    return (short_mutual_info)


def opt_step(question_set, product_set, traffic_set, purchased_set, use_history = False, df_history = 0, alpha = 2):
    """Maximal mutual information greedy step to select the best questions to ask given the history:
    Args:
        question_set: set of questions already asked
        product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: purchased table [	ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        use_history (default = False): boolean, if True then Entropy will be multiplied by the history probability of filters
        df_history (default = 0): history table obtained with algo_utils.create_history(traffic_cat, question_text_df)
                                  [ProductId	text	frequency]
        alpha (default = 2): parameter to determine the importance of history data, the higher the more important history is
    Returns:
        next_question: questionId to ask
     """
    MI_matrix = np.zeros([len(question_set), 2])
    mutual_array = np.asarray(parmap.map(mutual_inf, \
                                        question_set, \
                                        product_set=product_set, \
                                        traffic_set=traffic_set, \
                                        purchased_set=purchased_set,
                                        pm_pbar=True,
                                        pm_parallel=True,
                                        pm_processes = multiprocessing.cpu_count()))
    MI_matrix[:,0] = list(question_set)

    #Prior on filters already been used by users (more user-friendly)
    if use_history:
        Q_distr = algo_utils.get_proba_Q_distribution(list(question_set), df_history, alpha)
        MI_matrix[:, 1] = np.multiply(mutual_array, Q_distr)
    else:
        MI_matrix[:,1] = mutual_array
    
    next_question_index = np.argmax(MI_matrix, axis=0)[1] 
    next_question = MI_matrix[next_question_index, 0]
    print(next_question)
    return int(next_question)


def max_info_algorithm(product_set, traffic_set, purchased_set, \
                       question_text_df, answer_text_df, threshold, \
                       y, answers_y, use_history = False, df_history = 0, \
                       alpha = 2, first_questions = None):
    """Maximan mutual information algorithm to select the best subset of questions to ask:
    Args:
        product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
        purchased_set: purchased table [	ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        question_text_df: table to link questionId to text [PropertyDefinition	PropertyDefinitionId]
        answer_text_df: table to link answerId to text [answer_id	question_id	answer_text]
        threshold: max length of final set of products
        y: product selected for the algorithm
        answers_y: dict of question: np.array(answers), obtained with function: sample_answers(y, products_cat)
        use_history (default = False): boolean, True: use history data
        df_history (default = 0): history table obtained with algo_utils.create_history(traffic_cat, question_text_df)
                                  [ProductId	text	frequency]
        alpha (default = 2): parameter to determine the importance of history data, the higher the more important history is
        first_questions (default = None): optimization step, precompute the firsts questions, create new if there are none
    Returns:
        final_question_list: sequence of questionId to ask
        product_set: final product list
        y: product chosen as input of algo
        final_question_text_list:  sequence of questionText to ask
        answer_text_list: answers for each final question
     """
    question_set = set(algo_utils.get_questions(product_set))
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    distinct_products = product_set.ProductId.unique()
    print("There are {} questions we can ask".format(len(question_set)))
    print("There are {} possible products to choose from".format(len(distinct_products)))
    iter = 1

    #Compute the first 3 optimized questions for IDK answers (speed-up)
    if first_questions is None:
        first_questions = []
        first_question_set = question_set
        n_first_q = 3 
        print("Optimization: computing first {} questions".format(n_first_q))
        for i in range(n_first_q):
            first_question = opt_step(first_question_set, product_set, traffic_set, purchased_set, use_history, df_history, alpha)
            first_questions.append(first_question)
            first_question_set = first_question_set.difference(set(first_questions))

    #Given we have the first 3 best questions for IDK answer we can use them until we receive a different answer
    n_first_q = len(first_questions)
    idk = True
    i = 0
    while(idk and i < n_first_q):
        next_question = first_questions[i]
        i += 1
        print("Next question is filter : {}".format(next_question))
        question_text = question_id_to_text(next_question, question_text_df)
        print("Question is: {}".format(question_text))
        final_question_list.append(int(next_question))
        final_question_text_list.append(question_text)
        answer = answers_y.get(next_question)
        if not answer == ["idk"]:
            idk = False
        answer_text = answer_id_to_text(answer, next_question, answer_text_df)
        print("Answer given was: {}".format(answer))
        print("Answer was: {}".format(answer_text))
        answer_text_list.append(answer_text)
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=next_question, answer=answer, product_set=product_set, traffic_set =traffic_set, purchased_set = purchased_set)
        question_set_new = set(product_set["PropertyDefinitionId"].values)
        question_set = question_set_new.difference(final_question_list)
        distinct_products = len(product_set.ProductId.unique()) # faster
        print("There are {} more questions we can ask".format(len(question_set)))
        print("There are {} possible products to choose from".format(distinct_products))
        iter+=1

    #Perform greedy step until the subset of products is smaller than a certain threshold
    while not (distinct_products < threshold or len(question_set) == 0):
        next_question = opt_step(question_set, product_set, traffic_set, purchased_set, use_history, df_history, alpha)
        print("Next question is filter : {}".format(next_question))
        question_text = question_id_to_text(next_question, question_text_df)
        print("Question is: {}".format(question_text))
        final_question_list.append(int(next_question))
        final_question_text_list.append(question_text)
        answer = answers_y.get(next_question)
        answer_text = answer_id_to_text(answer, next_question, answer_text_df)
        print("Answer given was: {}".format(answer))
        print("Answer was: {}".format(answer_text))
        answer_text_list.append(answer_text)    
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=next_question, answer=answer, product_set=product_set, traffic_set =traffic_set, purchased_set = purchased_set)
        question_set_new = set(product_set["PropertyDefinitionId"].values) 
        question_set = question_set_new.difference(final_question_list)
        distinct_products = len(product_set.ProductId.unique()) # faster
        print("There are {} more questions we can ask".format(len(question_set)))
        print("There are {} possible products to choose from".format(distinct_products))
        iter+=1
    return final_question_list, product_set, y, final_question_text_list, answer_text_list


if __name__=='__main__':

    #upload all the data tables
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        question_text_df = load_obj('../data/question_text_df')
        answer_text_df = load_obj('../data/answer_text')
        print("Loaded datsets")
    #download all the data tables
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

    #upload history from traffic_cat
    try:
        df_history = load_obj('../data/df_history')
    #download history from traffic_cat
    except:
        df_history = algo_utils.create_history(traffic_cat, question_text_df)
        save_obj(df_history, '../data/df_history')
        print("Created history")

    #select a product
    y = products_cat["ProductId"][10]
    answers_y = sample_answers(y, products_cat)
    threshold = 50
    start_time = time.time()
    print("Start time: {}".format(start_time))
    #algorithm
    final_question_list, product_set, y, final_question_text_list, answer_text_list = max_info_algorithm(products_cat, \
                                                                                                         traffic_cat, \
                                                                                                         purchased_cat, \
                                                                                                         question_text_df, \
                                                                                                         answer_text_df, \
                                                                                                         threshold, y, \
                                                                                                         answers_y, \
                                                                                                         use_history = True, \
                                                                                                         df_history = df_history,
                                                                                                         alpha = 2)
    end_time = time.time()
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(product_set.ProductId.unique()))
    print("The algorithm took {}s.".format(end_time-start_time))
