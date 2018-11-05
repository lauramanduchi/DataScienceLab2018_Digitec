import numpy as np
import pandas as pd
import algo_utils
import random
import os
from load_utils import *
from RandomBaseline import  get_distinct_products
from init_dataframes import init_df


def get_all_answers(question, product_set):
    return(product_set.loc[product_set["PropertyDefinitionId"]==question, 'answer'].drop_duplicates().values)

def sample_answers(y, product_set, p_idk = 0.1, p_2a = 0.3, p_3a = 0.15):
    "returns a dict of question: np.array(answers)"
    question_set = set(algo_utils.get_questions(product_set))
    quest_answer_y = algo_utils.get_answers_y(y, product_set)
    result = {}
    for question in question_set:
        u = random.random()
        if u < p_idk:
           result[question] = ['idk'] 
        elif quest_answer_y[question]=='none': #if none you can't have a 2nd answer
            result[question] = [quest_answer_y[question]]
        elif u < p_idk+p_2a:
            possible = get_all_answers(question, product_set)
            sample = np.random.choice(possible, size=1)
            #print('orig {}'.format(quest_answer_y[question]))
            while (str(quest_answer_y[question]) in sample.astype(str)): #we want to sample ANOTHER
                sample = np.random.choice(possible, size=1)
            result[question] = np.append([quest_answer_y[question]], sample)
            #print('new {}'.format(result[question]))
        elif u < p_idk+p_2a+p_3a:
            possible = get_all_answers(question, product_set)
            sample = np.random.choice(possible, size=2, replace=False)
            #print('orig {}'.format(quest_answer_y[question]))
            while (str(quest_answer_y[question]) in sample.astype(str)): #we want to sample ANOTHER
                sample = np.random.choice(possible, size=2)
            result[question] = np.append([quest_answer_y[question]], sample)
            #print('new {}'.format(result[question]))
        else:
            result[question] = [quest_answer_y[question]] 
    return(result)

if __name__=='__main__':
    try:
        products_cat = load_obj('../data/products_table')
        print("Loaded datsets")
    except:
        print("Creating datasets...")
        products_cat, traffic_cat, purchased_cat = init_df()
  
    y = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = 1)[0]
    print(sample_answers(y, products_cat))