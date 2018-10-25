'''
# At every answer node, Compute I(X_e,Y;phi_l) and Choose X_e tlq Mututal Information is maximized
'''
from load_utils import *
import random
import numpy as np
import pandas as pd
#from utils import *
from init_dataframes import init_df
import algo_utils



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
def random_baseline(product_set, traffic_set, purchased_set, threshold, y):
    question_set = set(algo_utils.get_questions(product_set))
    print(question_set)
    quest_answer_y = algo_utils.get_answers_y(y, product_set)
    final_question_list=[]

    distinct_products = get_distinct_products(product_set)   #laura

    while not (len(distinct_products) < threshold or len(question_set) == 0):     #laura
        next_question = np.random.choice(np.asarray(list(question_set)), size=1)[0]
        final_question_list.append(int(next_question))
        answer = quest_answer_y[int(next_question)]          #laura
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=int(next_question), answer=str(answer), product_set=product_set,traffic_set =traffic_set, purchased_set = purchased_set)
        question_set_new = set(algo_utils.get_filters_remaining(product_set))
        question_set = question_set_new.difference(final_question_list) # s- t is written s.difference(t)
        distinct_products = get_distinct_products(product_set) #laura
        print(len(get_distinct_products(product_set)))
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
    print(products_cat["ProductId"].dtype) #int
    print(products_cat["PropertyDefinitionId"].dtype) #int
    print(products_cat["answer"].dtype) #float
    final_question_list, product_set, y = random_baseline(products_cat, traffic_cat, purchased_cat, threshold, y)
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(get_distinct_products(product_set)))

