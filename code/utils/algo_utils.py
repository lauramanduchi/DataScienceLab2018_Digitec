#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Data Science Lab Project - FALL 2018
Mélanie Bernhardt - Mélanie Gaillochet - Laura Manduchi

This file defines all the helper functions for the maxMI algorithm.
"""

import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
import numpy as np

from utils.build_answers_utils import question_id_to_text


def create_history(traffic_cat, question_text_df):
    """ Create history dataframe of filters used with their frequency:
        Args:
            traffic_cat: traffic table [SessionId	answers_selected	Items_ProductId]
            question_text_df: table to link questionId to text [PropertyDefinition	PropertyDefinitionId]
        Returns:
            df_history: history dataframe of filters used [QuestionId	text	frequency]
    """
    # Compute the list of all the filters used in history
    list_filters_used = []
    [list_filters_used.append(k) for t in traffic_cat["answers_selected"] for k in t.keys()]
    unique_filters = set(list_filters_used)
    df_history = pd.DataFrame(columns=["questionId", "text", "frequency"])
    total_freq = 0
    for f in unique_filters:
        question_text = question_id_to_text(f, question_text_df)
        if not question_text == 'No text equivalent for question':
            freq = list_filters_used.count(f)
            total_freq += freq
            df_history.loc[len(df_history)] = [f, question_text, freq]
    df_history["frequency"] = df_history["frequency"] / total_freq
    return df_history

def select_subset(product_set, traffic_set = [], question = None, answer = None, purchased_set = []):
    """Select the products corresponding to the answer given a filter and restrict the corresponding tables.
       Call init_dataframe to get the right product_set and traffic_set!:
        Args:
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
            traffic_set: traffic table [SessionId	answers_selected	Items_ProductId]
            question: filter used
            answer: answer given the filter
            purchased_set: purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
        Returns:
            product_set: restricted product table
            traffic_set: restricted traffic table
            purchased_set: restricted purchased table or [] if input was default
    """
    all_products = set(product_set["ProductId"].values)
    if np.array_equal(["idk"], answer):
        # in case the user doesn't know the answer return everything
        return(product_set, traffic_set, [])
    else:
        answer = [str(x) for x in answer]
        tmp = product_set.loc[(product_set["PropertyDefinitionId"]==int(float(question))) & (product_set["answer"].astype(str).isin(answer)), ]
        products_to_keep = np.unique(tmp["ProductId"])
        product_set = product_set.loc[product_set["ProductId"].isin(products_to_keep),]
        if len(traffic_set) != 0:
            traffic_set = traffic_set.loc[traffic_set["Items_ProductId"].isin(products_to_keep),]
        if len(purchased_set) != 0:
            purchased_set = purchased_set.loc[purchased_set["Items_ProductId"].isin(products_to_keep),]
        if len(products_to_keep)==0:
            print('problem')
            print(len(all_products))
        return(product_set, traffic_set, purchased_set)

def get_proba_Y_distribution(products_cat, purchased_cat, alpha=1):
    """Compute the probability of the products according to history:
        Args:
            products_cat: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
            purchased_cat: purchased table [ProductId	UserId	OrderId	SessionId	Items_ProductId	Items_ItemCount]
            alpha: alpha = 0 means uniform distribution for all the products, otherwise the bigger it is the more history is taken into account
        Returns:
            distribution: probability of the products according to history
    """
    # step 1 compute uniform distribution
    distribution = pd.DataFrame()
    unique_ids = products_cat['ProductId'].drop_duplicates().values
    number_prod_category_6 = len(unique_ids)
    proba_u = 1.0/number_prod_category_6 #uniform distribution: each product has the same probability of being bought by the client
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
    
    # step 3 add uniform and history*alpha and renormalize to get a probability
    unormalized_final_proba = distribution["uniform"].values + alpha*distribution["proportion_sold"].values 
    distribution["final_proba"] = unormalized_final_proba/np.sum(unormalized_final_proba)
    return(distribution)


def get_questions(product_set):
    """Compute all possible filters used in product_set:
        Args:
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        Returns:
            questions: available questions (filters) given the product table
    """
    return(product_set["PropertyDefinitionId"].drop_duplicates().values)

def get_answers_y(y, product_set):
    """Compute dictionary of {question: answers} given a product y:
        Args:
            y: given product for which compute the question-answers dictionary
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
        Returns:
            res: dictionary of {question: answers} for given product y
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


def get_proba_Q_distribution(question_list, df_history, alpha):
    """Compute the probability of the filters according to history:
        Args:
            question_list: list of questions available
            df_history: history dataframe of filters used [QuestionId	text	frequency] (see function create_history)
            alpha: alpha = 0 means uniform distribution for all the filters, otherwise the bigger it is the more history is taken into account
        Returns:
            Q_proba: probability distribution of filters (questions)
    """
    n = len(question_list)
    # Step 1: uniform distribution
    Q_proba = np.ones(n)/n
    # Step 2: taking into account history
    for i in range(n):
        q_id = str(int(float(question_list[i])))
        try:
            Q_proba[i] += alpha * df_history["frequency"].loc[df_history["questionId"] == q_id].values[0]
        except IndexError:
            pass
    # Step 3: renormalizing
    Q_proba = Q_proba / Q_proba.sum()
    return Q_proba

def get_proba_A_distribution_none(question, products_cat, traffic_processed, alpha=1):
    """Compute the probability of answers given a question according to the remaining products:
        Args:
            question: selected question, the algorithm computes the probabilities of its possible answers
            product_set: product table [ProductId	BrandId	ProductTypeId	PropertyValue	PropertyDefinitionId	PropertyDefinitionOptionId	answer]
            traffic_processed: traffic table [SessionId	answers_selected	Items_ProductId]
            alpha: alpha = 0 means fraction of products per answer without the history data, otherwise the bigger it is the more history is taken into account
        Returns:
            distribution: probability distribution of answers given a question
    """
    distribution = pd.DataFrame()
    number_products_total = len(products_cat['ProductId'].drop_duplicates().values)
    
    if (number_products_total==0):
        print('Nothing to return there is no product left with this filter')
        return(distribution)
    
    # step 1: probas is number of product per answer to the question (no history)
    products_cat = products_cat.loc[products_cat["PropertyDefinitionId"]==int(question), ]
    nb_prod_with_answer = len(np.unique(products_cat["ProductId"]))
    distribution["nb_prod"] = products_cat[['ProductId','answer']].groupby(['answer']).count()["ProductId"]
    distribution.index = distribution.index.astype(float)
    nb_prod_without_answer = number_products_total - nb_prod_with_answer
    distribution["catalog_proba"] = distribution["nb_prod"]/number_products_total
    
    # step 2: add the history if available just for KNOWN answers
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

    #step 3: combine the two probabilities as: p1 + alpha * p2
    distribution["final_proba"] = distribution["catalog_proba"].values + alpha*distribution["history_proba"].values

    #step 4: add the idk answer and relative probability JUST FROM CATALOG
    if nb_prod_without_answer!=0:
        distribution.loc["idk", "final_proba"] = nb_prod_without_answer/float(number_products_total)

    #step 5: renormalize everything
    distribution["final_proba"] = distribution["final_proba"]/distribution["final_proba"].sum()
    return(distribution)