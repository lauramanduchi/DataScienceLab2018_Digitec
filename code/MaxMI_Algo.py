'''
# At every answer node, Compute I(X_e,Y;phi_l) and Choose X_e tlq Mututal Information is maximized
'''
from load_utils import *
import random
import numpy as np
import pandas as pd
from init_dataframes import init_df
import algo_utils
import parmap
import time

# about parmap 
#y = [myfunction(x, argument1, mykeyword=argument2) for x in mylist]
# In parallel:
#y = parmap.map(myfunction, mylist, argument1, mykeyword=argument2)


'''Normalized (# products given answer/total # products given question)'''
def prob_answer(answer, question, product_set, traffic_set):
    p_answer = algo_utils.get_proba_Q_distribution(question, product_set, traffic_set, alpha=1) #Mel function name
    return p_answer.loc[float(answer)]["final_proba"]


'''Normalized (# products y bought/total #products bought) + (# products y /total #products)'''
def prob_product(product, product_set, purchased_set):
    p_product = algo_utils.get_proba_Y_distribution(product_set, purchased_set, alpha=1) #Mel function name
    return p_product.loc[product]["final_proba"]

'''
Updates product_set, traffic_set and purchased_set to account for conditionality
'''
def conditional_entropy(answer, question, product_set, traffic_set, purchased_set): #laura purchased_set deleted
    product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=answer,
                                                            product_set=product_set, traffic_set =traffic_set,
                                                            purchased_set = purchased_set)
    product_ids = product_set["ProductId"].drop_duplicates().values
    cond_entropy_y = 0
    p_product_given_a = algo_utils.get_proba_Y_distribution(product_set, purchased_set, alpha=1)["final_proba"]
    for product in product_ids:
        prob_y_given_a = p_product_given_a.loc[product]
        cond_entropy_y += prob_y_given_a * np.log(prob_y_given_a)
    """
    for product in product_ids:
        prob_y_given_a = prob_product(product, product_set, purchased_set)
        cond_entropy_y += prob_y_given_a * np.log(prob_y_given_a)
    """
    return cond_entropy_y


def mutual_inf(question, product_set, traffic_set, purchased_set):
    short_mutual_info = 0
    answer_set = product_set.loc[product_set["PropertyDefinitionId"]==question, "answer"].drop_duplicates().values
    p_answer = algo_utils.get_proba_Q_distribution(question, product_set, traffic_set, alpha=1)["final_proba"]
    product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=None,
                                                            product_set=product_set, traffic_set = traffic_set,
                                                            purchased_set = purchased_set)
    for answer in answer_set:
        short_mutual_info += - p_answer.loc[float(answer)]* \
                             conditional_entropy(answer, None, product_set, traffic_set, purchased_set)
    print(short_mutual_info)
    return short_mutual_info


# Return question which maximizes MI
def opt_step(question_set, product_set, traffic_set, purchased_set):
    MI_matrix = np.zeros([len(question_set), 2])
    mutual_array = np.asarray(parmap.map(mutual_inf, question_set, product_set=product_set, traffic_set=traffic_set, purchased_set=purchased_set))
    MI_matrix[:,0] = list(question_set)
    MI_matrix[:,1] = mutual_array
    next_question_index = np.argmax(MI_matrix, axis=0)[1]
    next_question = MI_matrix[next_question_index, 0]
    print(next_question)
    return next_question

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
def max_info_algorithm(product_set, traffic_set, purchased_set, threshold, y):
    question_set = set(algo_utils.get_questions(product_set))
    quest_answer_y = algo_utils.get_answers_y(y, product_set)
    final_question_list=[]
    distinct_products = get_distinct_products(product_set)   #laura
    print("There are {} questions we can ask".format(len(question_set)))
    print("There are {} possible products to choose from".format(len(distinct_products)))
    iter = 1
    while not (len(distinct_products) < threshold or len(question_set) == 0):     #laura
        print("Processing question: {}".format(iter))
        next_question = opt_step(question_set, product_set, traffic_set, purchased_set)
        print("Next question is filter : {}".format(next_question))
        final_question_list.append(next_question)
        answer = quest_answer_y[int(next_question)][0]          #laura
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=next_question, answer=answer, product_set=product_set, traffic_set =traffic_set, purchased_set = purchased_set)
        question_set_new = set(algo_utils.get_filters_remaining(product_set))
        question_set = question_set_new.difference(final_question_list)
        distinct_products = get_distinct_products(product_set) #laura
        print("There are {} more questions we can ask".format(len(question_set)))
        print("There are {} possible products to choose from".format(len(get_distinct_products(product_set))))
        iter+=1
    return final_question_list, product_set, y



if __name__=='__main__':
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        print("Loaded datsets")
    except:
        print("Creating datasets...")
        products_cat, traffic_cat, purchased_cat = init_df()
    
    y = products_cat["ProductId"][10]
    threshold = 50
    start_time = time.time()
    print("Start time: {}".format(start_time))
    final_question_list, product_set, y = max_info_algorithm(products_cat, traffic_cat, purchased_cat, threshold, y)
    end_time = time.time()
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(product_set))
    print("The algorithm took {}s.".format(end_time-start_time))
