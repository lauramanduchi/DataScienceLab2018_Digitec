import numpy as np
import pandas as pd

import random
import os
from utils.load_utils import *
import utils.algo_utils as algo_utils

""" 
This file defines the answer sampler.
"""

def get_all_answers(question, product_set):
    """ Returns all possible answers to a given questions.
    Args:
        question(int): id of question
        product_set: input product catalog dataframe 
    Returns:
        List of possible answers
    """
    return(product_set.loc[product_set["PropertyDefinitionId"]==question, 'answer'].drop_duplicates().values)

def sample_answers(y, product_set, p_idk = 0.1, p_2a = 0.3, p_3a = 0.15):
    """ This is the answers sampler given a target product.
    Args:
        y: target product for which you want to sample the answers
        product_set: product catalog in which the answers can be found
        p_idk (default 0.1): additional probability of answering "I dont know" to a given question
        p_2a (default 0.3): probability of giving 2 answers to a given question (true answer and random other one)
        p_3a (default 0.15): probability of giving 3 answers to a given question.
    
    Returns:
        result(dict): dictionary where the key is question and 
                      the value is a np.array containing the 
                      answers to this question.
    """
    # Get set of possible questions available in the product catalog
    question_set = set(product_set["PropertyDefinitionId"].values)  # faster
    
    # Get dict of (true) answers available for the target product
    quest_answer_y = algo_utils.get_answers_y(y, product_set) 
    result = {}
    
    # For each question sample additional answers 
    # or replace true answer by idk if necessary.
    for question in question_set:
        # Sample random number b/w 0 and 1.
        u = random.random()
        # Sample if user says idk
        if u < p_idk:
           result[question] = ['idk'] 
        # Else if it is possible sample if user give additional answers.
        elif quest_answer_y[question]=='none': #if none you can't have a 2nd answer
            result[question] = [quest_answer_y[question]]
        elif quest_answer_y[question]=='idk': #if none you can't have a 2nd answer
            result[question] = [quest_answer_y[question]]        
        # Giving 2 answers?
        elif u < p_idk+p_2a:
            possible = get_all_answers(question, product_set)
            sample = np.random.choice(possible, size=1)
            # If the drawn 2nd answer is the same, redraw one
            while (str(quest_answer_y[question]) in sample.astype(str)): 
                sample = np.random.choice(possible, size=1)
            result[question] = np.append([quest_answer_y[question]], sample)   
        # Giving 3 answers?
        elif u < p_idk+p_2a+p_3a:
            possible = get_all_answers(question, product_set)
            sample = np.random.choice(possible, size=2, replace=False)
            # If the drawn 2nd or 3rd answer is the same, redraw 2 answers
            while (str(quest_answer_y[question]) in sample.astype(str)):
                sample = np.random.choice(possible, size=2)
            result[question] = np.append([quest_answer_y[question]], sample)
        # Else keep only the true answer 
        else:
            result[question] = [quest_answer_y[question]] 
    return(result)





# =========== TESTING FUNCTION ========== #
if __name__=='__main__':
    from init_dataframes import init_df
    try:
        products_cat = load_obj('../data/products_table')
        print("Loaded datsets")
    except:
        print("Create datasets first")   
    y = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = 1)[0]
    print(sample_answers(y, products_cat))