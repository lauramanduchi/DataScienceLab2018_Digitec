#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
import numpy as np
from utils.build_answers_utils import question_id_to_text


def create_history_old(traffic_cat, question_text_df):
    # Compute the list of all the filters used in history
    list_filters_used = []
    i = 0
    for t in traffic_cat["answers_selected"]:
        i += 1
        for k in t.keys():
            list_filters_used.append(k)
    unique_filters = set(list_filters_used)
    question_text_list = []
    df_history = pd.DataFrame(columns=["ProductId", "text", "frequency"])
    total_freq = 0
    for f in unique_filters:
        question_text = question_id_to_text(f, question_text_df)
        if not question_text == 'No text equivalent for question':
            question_text_list.append(question_text)
            freq = list_filters_used.count(f)
            total_freq += freq
            df_history.loc[len(df_history)] = [f, question_text, freq]
    df_history["frequency"] = df_history["frequency"] / total_freq
    return df_history

def create_history(traffic_cat, question_text_df):
    """old took 0.07480597496032715
       new took 0.027109861373901367
       gain: 2.76 speed up"""
    # Compute the list of all the filters used in history
    list_filters_used = []
    for t in traffic_cat["answers_selected"]:
        list_filters_used += [k for k in t.keys()]
    unique_filters = set(list_filters_used)
    question_text_list = []
    df_history = pd.DataFrame(columns=["ProductId", "text", "frequency"])
    total_freq = 0
    for f in unique_filters:
        question_text = question_id_to_text(f, question_text_df)
        n=0
        if not question_text == 'No text equivalent for question':
            question_text_list.append(question_text)
            freq = list_filters_used.count(f)
            total_freq += freq
            n += 1
            df_history.loc[n] = [f, question_text, freq]
    df_history["frequency"] = df_history["frequency"] / total_freq
    return df_history


def get_distinct_products(product_set):
    try:
        distinct_p = product_set.ProductId.unique()
    except AttributeError:
        print("'ProductId' is not a valid column in Product_set, rename it!")
    return distinct_p

def select_subset(product_set, traffic_set = [], question = None, answer = None, purchased_set = []):
    """
    function assumes you have already build the answer column
    """
    all_products = set(product_set["ProductId"].values)
    if np.array_equal(["idk"], answer): # case i don't know the answer return everything
        return(product_set, traffic_set, [])
    else:
        q_keep = set(product_set.loc[product_set["PropertyDefinitionId"]==int(question), "ProductId"].drop_duplicates().values) # had to remove this drop_duplicates() because it changed the index !!!!!!!
        if np.array_equal(["none"], answer):
            products_to_keep = q_keep
        else:
            a_keep = set()
            for a in answer:
                a_keep = a_keep.union(set(product_set.loc[product_set["answer"].astype(str)==str(a), "ProductId"].drop_duplicates().values))
            products_to_keep = a_keep.intersection(q_keep)
        if np.array_equal(["none"], answer):
            products_to_keep = all_products.difference(products_to_keep)
        product_set = product_set.loc[product_set["ProductId"].isin(products_to_keep),]
        if len(traffic_set) != 0:
            traffic_set = traffic_set.loc[traffic_set["Items_ProductId"].isin(products_to_keep),]
        if len(purchased_set) != 0:
            purchased_set = purchased_set.loc[purchased_set["Items_ProductId"].isin(products_to_keep),]
        if len(get_distinct_products(product_set))==0:
            print('problem')
            print(len(q_keep))
            print(len(all_products))
        return(product_set, traffic_set, purchased_set)

def get_proba_Y_distribution(products_cat, purchased_cat, alpha=1):
    distribution = pd.DataFrame()
    unique_ids = products_cat['ProductId'].drop_duplicates().values
    number_prod_category_6 = len(unique_ids)
    proba_u = 1.0/number_prod_category_6 # if all products had the same probability to be bought
    distribution["uniform"] = np.repeat(proba_u, number_prod_category_6)
    distribution.index = unique_ids
    
    distribution["proportion_sold"] = 0.0 # init to 0
    # step 2 take history into accounts
    if len(purchased_cat) > 0:
        sold_by_product = purchased_cat.groupby('ProductId').sum()["Items_ItemCount"]
        prod_ids = sold_by_product.index.values
        total_sold = np.sum(sold_by_product.values)
        adjust_proba_by_product = sold_by_product.values/float(total_sold)
        distribution.loc[prod_ids, "proportion_sold"] = adjust_proba_by_product
    
    # step 3 add uniform and history and renormalize to get a proba
    unormalized_final_proba = distribution["uniform"].values + alpha*distribution["proportion_sold"].values 
    distribution["final_proba"] = unormalized_final_proba/np.sum(unormalized_final_proba)
    return(distribution)


def get_questions(product_set):
    return(product_set["PropertyDefinitionId"].drop_duplicates().values)

def get_answers_y_old(y, product_set):
    """ can be removed """ 
    questions = get_questions(product_set)
    tmp = product_set.loc[product_set["ProductId"]==y,]
    res = {}
    for q in questions:
        a = tmp.loc[product_set["PropertyDefinitionId"]==int(q),"answer"].values
        if len(a)==0:
            #res.update({q: 'none'}) #TODO discuss on thursday to be sure what we do
            res.update({q: 'idk'})
        else:
            if np.isnan(a[0]):
                res.update({q: 'idk'})
            else:
                res.update({q: a[0]})
    return(res)

def get_answers_y(y, product_set):
    """ speed up check
    new took 0.04408001899719238
    old took 0.11850404739379883
    gain: time divided by 2.75
    """
    questions = product_set["PropertyDefinitionId"].drop_duplicates().values # supposed to be faster
    tmp = product_set.loc[product_set["ProductId"]==y, ["answer", "PropertyDefinitionId"]]
    res = {}
    for q in questions:
        try:
            a = tmp.loc[tmp["PropertyDefinitionId"]==int(q),"answer"].values[0] # way faster
            if np.isnan(a):
                res.update({q: 'idk'})
            else:
                res.update({q: a})
        except IndexError:
            res.update({q: 'idk'})
    return(res)

def get_filters_remaining(dataset):
    return(dataset["PropertyDefinitionId"].drop_duplicates().values)

def get_proba_Q_distribution(question_list, df_history, alpha):
    Q_proba = np.zeros(len(question_list))
    for i in range(len(question_list)):
        q_id = str(int(question_list[i]))
        Q_proba[i] = 1 / len(question_list)
        if q_id in df_history["ProductId"].values:
            Q_proba[i] += alpha * df_history["frequency"].loc[df_history["ProductId"] == q_id].values[0]
    Q_proba = Q_proba / Q_proba.sum()
    return Q_proba
    

def get_proba_A_distribution_none(question, products_cat, traffic_processed, alpha=1):
    """
    assumes answer is already constructed
    """
    distribution = pd.DataFrame()
    number_products_total = len(products_cat['ProductId'].drop_duplicates().values)
    if (number_products_total==0):
        print('Nothing to return there is no product left with this filter')
        return(distribution)
     # step 1: probas is number of product per answer to the question (no history)
    possible_answers = products_cat.loc[products_cat["PropertyDefinitionId"]==int(question), "answer"] \
                                    .drop_duplicates().values.astype(float)
    nb_prod_per_answer = []
    for a in possible_answers:
        nb_prod_per_answer.append(len(select_subset(products_cat, [], question, np.asarray([a]))[0]["ProductId"].drop_duplicates().values))
    distribution["nb_prod"] = nb_prod_per_answer
    distribution.index = possible_answers #type float64
    nb_prod_with_answer = len(products_cat.loc[products_cat["PropertyDefinitionId"]==int(question), "ProductId"].drop_duplicates().values) # new
    nb_prod_without_answer = number_products_total - nb_prod_with_answer
    distribution["catalog_proba"] = np.asarray(nb_prod_per_answer)/float(number_products_total) # new
    #step 2: add the history if available just for KNOWN answers
    distribution["history_proba"] = 0
    if (len(traffic_processed)>0):
        history_answered = []
        response = traffic_processed["answers_selected"].values
        for r_dict in response:
            if str(question) in r_dict:
                history_answered.extend(r_dict[str(question)])
        if not history_answered == []: 
            series = pd.Series(history_answered)
            add_probas = series.value_counts()
            s_add = sum(add_probas.values)
            add_probas = add_probas/s_add
            index = add_probas.index
            for i in index:
                if float(i) in distribution.index:
                    distribution.loc[float(i), "history_proba"] = add_probas.loc[i]
    distribution["final_proba"] = distribution["history_proba"].values + alpha*distribution["catalog_proba"].values
    # add the idk case JUST FROM CATALOG
    if nb_prod_without_answer!=0: # Only add None if it is a POSSIBLE ANSWER!!
        distribution.loc["idk", "final_proba"] = nb_prod_without_answer/float(number_products_total)
    # renormalize everything
    S = np.sum(distribution["final_proba"].values)
    distribution["final_proba"] = distribution["final_proba"]/S
    #print(distribution)
    return(distribution)


if __name__ == "__main__":
    from utils.load_utils import *
    from utils.init_dataframes import init_df
    import utils.algo_utils as algo_utils
    from utils.build_answers_utils import question_id_to_text, answer_id_to_text
    from utils.sampler import sample_answers
    import time 

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

    #uploading history from traffic_cat
    try:
        df_history = load_obj('../data/df_history')
    except:
        df_history = algo_utils.create_history(traffic_cat, question_text_df)
        save_obj(df_history, '../data/df_history')
        print("Created history")


    y = products_cat["ProductId"][20]
    answers_y = sample_answers(y, products_cat)
    threshold = 50
    start_time = time.time()
    get_answers_y(y, products_cat)
    print ('new took {}'.format(time.time()-start_time))
    start_time = time.time()
    get_answers_y_old(y, products_cat)
    print ('old took {}'.format(time.time()-start_time))
    
    start_time = time.time()
    create_history_old(traffic_cat, question_text_df)
    print ('old took {}'.format(time.time()-start_time))
    start_time = time.time()
    create_history(traffic_cat, question_text_df)
    print ('new took {}'.format(time.time()-start_time))

