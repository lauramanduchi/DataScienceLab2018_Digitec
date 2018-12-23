""" Data Science Lab Project - FALL 2018
Mélanie Bernhardt - Mélanie Gaillochet - Laura Manduchi

This file defines the randome baseline used in evaluation.
"""
import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import random
import numpy as np
import pandas as pd

from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
import utils.build_answers_utils as build_answers_utils
import utils.sampler as sampler


def random_baseline(product_set, traffic_set, purchased_set, 
                    question_text_df, answer_text_df,
                    threshold, y, answers_y):
    """Random baseline algorithm.
    Note: 
        at each timestep sample randomly one question among the remaining questions.

    Args:
        product_set: the initial set product available (catalog)
        traffic_set: the initial traffic dataset
        purchased_set: the initial purchased dataset
        question_text_df: dataframe containing the correspondance between questionIds and string.
        answer_text_df: dataframe containing the correspondance between answerIds and string.
        threshold: stopping criteria of the algorithm
        y: target product
        answers_y: sampled answers for target product

    Returns:
        final_question_list: final list of asked questionsIds
        product_set: final set of selected products
        y: target product
        final_question_text_list: final list of asked question (string)
        answer_text_list: final list of given answers (as string)
    """
    question_set = set(product_set["PropertyDefinitionId"].values)
    quest_answer_y = answers_y
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    distinct_products = len(product_set.ProductId.unique()) # faster   
    while not (distinct_products < threshold or len(question_set) == 0):   
        next_question = np.random.choice(np.asarray(list(question_set)), size=1)[0]
        next_question = int(next_question)
        print("RDM: Next question is filter : {}".format(next_question))
        question_text = build_answers_utils.question_id_to_text(next_question, question_text_df)
        print("RDM: Question is: {}".format(question_text))
        final_question_list.append(int(next_question))
        final_question_text_list.append(question_text)
        answer = quest_answer_y[int(next_question)]
        answer_text = build_answers_utils.answer_id_to_text(answer, next_question, answer_text_df)
        print("RDM: Answer given was: {}".format(answer))
        print("RDM: Answer was: {}".format(answer_text))
        product_set, traffic_set, purchased_set = algo_utils.select_subset(
                                                        question=int(next_question),
                                                        answer=answer,
                                                        product_set=product_set,
                                                        traffic_set=traffic_set,
                                                        purchased_set = purchased_set)
        question_set_new = set(product_set["PropertyDefinitionId"].values) 
        question_set = question_set_new.difference(final_question_list)
        distinct_products = len(product_set.ProductId.unique())
        print('RDM: There are still {} products to choose from'.format(distinct_products))
    return final_question_list, product_set, y, final_question_text_list, answer_text_list



if __name__=='__main__':
    """ Just for testing purposes; """
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        question_text_df = load_obj('../data/question_text_df')
        answer_text_df = load_obj('../data/answer_text')
        print("Loaded datsets")
    except:
        print("Creating datasets...")
        products_cat, traffic_cat, purchased_cat = init_df()

    y = products_cat["ProductId"][10]
    threshold = 50
    print(products_cat["ProductId"].dtype) #int
    print(products_cat["PropertyDefinitionId"].dtype) #int
    print(products_cat["answer"].dtype) #float
    answers_y = sampler.sample_answers(y, products_cat)
    final_question_list, product_set, y, final_question_text_list, answer_text_list = random_baseline(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text_df, threshold, y, answers_y)
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(product_set.ProductId.unique()))

