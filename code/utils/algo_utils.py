#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

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
    if np.array_equal(["idk"], answer): # case i don't know the answer return everything
        return(product_set, traffic_set, [])
    else:
        q_keep = set(product_set.loc[product_set["PropertyDefinitionId"]==int(question), "ProductId"].index.values) # had to remove this drop_duplicates() because it changed the index !!!!!!!
        if np.array_equal(["none"], answer):
            total = q_keep
        else:
            a_keep = set()
            for a in answer:
                a_keep = a_keep.union(set(product_set.loc[product_set["answer"].astype(str)==str(a), "ProductId"].index.values))
            total = a_keep.intersection(q_keep)
        products_to_keep = product_set.loc[total, "ProductId"].drop_duplicates().values
        if np.array_equal(["none"], answer):
            product_set = product_set.loc[product_set["ProductId"].isin(products_to_keep)==False,]
            if len(traffic_set) != 0:
                traffic_set = traffic_set.loc[traffic_set["Items_ProductId"].isin(products_to_keep)==False,]
            if len(purchased_set) != 0:
                purchased_set = purchased_set.loc[purchased_set["Items_ProductId"].isin(products_to_keep)==False,]
        else:
            product_set = product_set.loc[product_set["ProductId"].isin(products_to_keep),]
            if len(traffic_set) != 0:
                traffic_set = traffic_set.loc[traffic_set["Items_ProductId"].isin(products_to_keep),]
            if len(purchased_set) != 0:
                purchased_set = purchased_set.loc[purchased_set["Items_ProductId"].isin(products_to_keep),]
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

def get_answers_y(y, product_set):
    questions = get_questions(product_set)
    tmp = product_set.loc[product_set["ProductId"]==y,]
    res = {}
    for q in questions:
        a = tmp.loc[product_set["PropertyDefinitionId"]==q,"answer"].values
        if len(a)==0:
            res.update({q: 'none'})
        else:
            res.update({q: a[0]})
    return(res)

def get_filters_remaining(dataset):
    return(dataset["PropertyDefinitionId"].drop_duplicates().values)
    

def get_proba_Q_distribution_none(question, products_cat, traffic_processed, alpha=1):
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
    nb_prod_without_answer = len(products_cat.loc[products_cat["PropertyDefinitionId"].isin([int(question)])==False, "ProductId"] \
                                    .drop_duplicates().values) # new
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
    distribution.loc["none", "final_proba"] = nb_prod_without_answer/float(number_products_total)
    #print(distribution["final_proba"])
    # renormalize everything
    S = np.sum(distribution["final_proba"].values)
    distribution["final_proba"] = distribution["final_proba"]/S
    return(distribution)