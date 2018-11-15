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


def conditional_entropy(answer, question, product_set, traffic_set, purchased_set): #laura purchased_set deleted
    product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=answer,
                                                            product_set=product_set, traffic_set =traffic_set,
                                                            purchased_set = purchased_set)
    product_ids = product_set["ProductId"].drop_duplicates().values
    cond_entropy_y = 0
    try:
        p_product_given_a = algo_utils.get_proba_Y_distribution(product_set, purchased_set, alpha=1)["final_proba"]
    except ZeroDivisionError:
        print('pbm only {} product left'.format(product_ids))
        print(answer)
        print(question)
    for product in product_ids:
        prob_y_given_a = p_product_given_a.loc[product]
        cond_entropy_y += prob_y_given_a * np.log(prob_y_given_a)
    return cond_entropy_y


def mutual_inf(question, product_set, traffic_set, purchased_set):
    short_mutual_info = 0
    proba_Q = algo_utils.get_proba_Q_distribution_none(question, product_set, traffic_set, alpha=1)["final_proba"]
    possible_answers = proba_Q.index
    for answer in possible_answers:
        short_mutual_info += proba_Q.loc[answer]* \
                             conditional_entropy(np.asarray([answer]), question, product_set, traffic_set, purchased_set) #TEST
    #print(short_mutual_info)
    return (short_mutual_info)


# Return question which maximizes MI
def opt_step(question_set, product_set, traffic_set, purchased_set):
    MI_matrix = np.zeros([len(question_set), 2])
    mutual_array = np.asarray(parmap.map(mutual_inf, \
                                        question_set, \
                                        product_set=product_set, \
                                        traffic_set=traffic_set, \
                                        purchased_set=purchased_set,
                                        pm_pbar=True,
                                        pm_parallel=True))
    MI_matrix[:,0] = list(question_set)
    MI_matrix[:,1] = mutual_array
    next_question_index = np.argmax(MI_matrix, axis=0)[1] 
    next_question = MI_matrix[next_question_index, 0]
    print(next_question)
    return int(next_question)

def get_distinct_products(product_set):
    try:
        distinct_p = product_set.ProductId.unique()
    except AttributeError:
        print("'ProductId' is not a valid column in Product_set, rename it!")
    return distinct_p


'''Our algorith which returns:
 1) sequence of question to ask
 2) final product list
 3) y chosen as input of algo'''
def max_info_algorithm(product_set, traffic_set, purchased_set, question_text_df, answer_text_df, threshold, y, answers_y):
    question_set = set(algo_utils.get_questions(product_set))
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    distinct_products = get_distinct_products(product_set)
    print("There are {} questions we can ask".format(len(question_set)))
    print("There are {} possible products to choose from".format(len(distinct_products)))
    iter = 1

    ## to update later
    """
    next_question = 522
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
    question_set_new = set(algo_utils.get_filters_remaining(product_set)) 
    question_set = question_set_new.difference(final_question_list)
    distinct_products = get_distinct_products(product_set)
    print("There are {} more questions we can ask".format(len(question_set)))
    print("There are {} possible products to choose from".format(len(get_distinct_products(product_set))))
"""
    iter+=1    
    while not (len(distinct_products) < threshold or len(question_set) == 0):
        next_question = opt_step(question_set, product_set, traffic_set, purchased_set)
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
        question_set_new = set(algo_utils.get_filters_remaining(product_set)) 
        question_set = question_set_new.difference(final_question_list)
        distinct_products = get_distinct_products(product_set)
        print("There are {} more questions we can ask".format(len(question_set)))
        print("There are {} possible products to choose from".format(len(get_distinct_products(product_set))))
        iter+=1
    return final_question_list, product_set, y, final_question_text_list, answer_text_list


if __name__=='__main__':
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        question_text_df = load_obj('../data/question_text_df')
        answer_text_df = load_obj('../data/answer_text')
        print("Loaded datsets")
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
    
    y = products_cat["ProductId"][10]
    answers_y = sample_answers(y, products_cat)
    threshold = 50
    start_time = time.time()
    print("Start time: {}".format(start_time))
    final_question_list, product_set, y, final_question_text_list, answer_text_list = max_info_algorithm(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text_df, threshold, y, answers_y)
    end_time = time.time()
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(get_distinct_products(product_set)))
    print("The algorithm took {}s.".format(end_time-start_time))
