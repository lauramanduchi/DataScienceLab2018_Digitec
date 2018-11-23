#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine,text
import pandas as pd
import numpy as np
import time

from utils.parser import parse_query_string
from utils.build_answers_utils import keep_only_useful_URLs, create_categories, eliminate_filters_no_answers, map_origAnswer_newAnswer, process_all_traffic_answers, map_text_new_answer
from utils.load_utils import load_obj, save_obj, batch
        
def init_df():
    # ------------------- DATABASE SETUP ------------- #
    # initialize connection to the machine
    engine = create_engine('postgresql://dslab:dslab2018@localhost/dslab')
    c = engine.connect()

    # ------------------- RELEVANT DATAFRAME SETUP ------------- #
    
    # ---------------  reduced purchased ---------------- #
    t1 = time.time()
    reduced_purchased = pd.read_sql_query(''' 
    SELECT "UserId", "OrderId", "SessionId", "Items_ProductId", "Items_ItemCount"
    FROM product_purchase
    WHERE ("OrderId" IN
                (SELECT "OrderId"
                FROM product_purchase 
                GROUP BY "OrderId" 
                HAVING count(distinct "Items_ProductId")=1)
            AND "SessionId" IS NOT NULL);
    ''', c)
    t2 = time.time()
    print('Created reduced_purchased in {}'.format(t2-t1))

    # Number of the category to use
    cat = 6

    # ---------------- Extract relevant products_cat ------------- #
    t1 = time.time()
    products_cat = pd.read_sql_query('''
    SELECT "ProductId", "BrandId", "ProductTypeId", "PropertyValue", "PropertyDefinitionId", "PropertyDefinitionOptionId"
    FROM product_only_ids
    WHERE "ProductTypeId"='{}' ;
    '''.format(cat), c)
    t2 = time.time()
    print('Created product_cat in {}s.'.format(t2-t1))
    print('Found {} items'.format(len(products_cat['ProductId'].drop_duplicates().values)))

    # ------------- Just productId (not sure we really need that) --- #
    t1 = time.time()
    productIdsCat = pd.read_sql_query('''
    SELECT DISTINCT "ProductId"
    FROM product_only_ids
    WHERE "ProductTypeId"='{}' ;
    '''.format(cat), c)
    t2 = time.time()
    print('Created productIdsCat in {}s.'.format(t2-t1))

    # -------- All products purchased from selected category ----- #
    t1 = time.time()
    purchased_cat = pd.merge(productIdsCat, reduced_purchased, \
                    left_on="ProductId", right_on="Items_ProductId", \
                    how = "inner")
    t2 = time.time()
    print('Created purchased_Cat in {}s.'.format(t2-t1))
    print('Found {} sold items. And {} unique product id'.format(len(purchased_cat), len(purchased_cat["ProductId"])))


    # --------- Extract of relevant traffic -------- #
    traffic_cat = pd.DataFrame()
    
    no_sessionId_found = 0
    no_matching_rows = 0
    SessionIds = purchased_cat["SessionId"].drop_duplicates().values.astype(int)
    no_sessionId_purchased = len(SessionIds)
    traffic_cat = pd.DataFrame()
    batch_size = 2000
    batches = batch(SessionIds, n=batch_size)

    i=0
    for b in batches:
        print("Processing batch: {}".format(i))
        i+=1
        s = text('''SELECT "RequestUrl", "Timestamp", "SessionId" FROM traffic 
                WHERE ("SessionId" IN {}
                 AND ("RequestUrl" LIKE '%opt%'
                 OR "RequestUrl" LIKE '%bra%'
                 OR "RequestUrl" LIKE '%pdo%'
                 OR "RequestUrl" LIKE '%rng%'))'''.format(tuple(b)))
        traffic_b = pd.read_sql(s, c)
        traffic_cat = traffic_cat.append(traffic_b, ignore_index= True)
        no_sessionId_found += len(traffic_b["SessionId"].drop_duplicates().values)
        no_matching_rows += len(traffic_b["SessionId"].values)
    print('Out of {} sessionsId found in the purchase dataset (category {}), {} were matched to at least one entry in the traffic table.'
          .format(no_sessionId_purchased, cat, no_sessionId_found))
    print('In total there were {} matching rows in the traffic dataset'.format(no_matching_rows))
    traffic_cat = keep_only_useful_URLs(traffic_cat)
    

    # ------------------- NEW ANSWERS SETUP - PRODUCT CATALOG ------------- #
    filters_def_dict, type_filters  = create_categories(products_cat)
    products_cat = eliminate_filters_no_answers(products_cat, type_filters)
    products_cat["answer"] = map_origAnswer_newAnswer(products_cat, filters_def_dict, type_filters)
    
    # ------------- Map traffic data to new answers --------------- #
    #traffic_cat = process_all_traffic_answers(traffic_cat, purchased_cat, filters_def_dict, type_filters)

    # ------------- get text of the question ------------- #
    question_text_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinition", "PropertyDefinitionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)
    print('Done question to text dataframe')
    # ---------- get text of answer ---------- #
    opt_answer_text_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinitionOption", "PropertyDefinitionOptionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)
    save_obj(opt_answer_text_df, '../data/opt_answer_text_df')

    print('Begin answer to text dataframe')
    answer_text = pd.DataFrame()
    answer_text["answer_id"] = products_cat["answer"]
    answer_text["question_id"] = products_cat["PropertyDefinitionId"]
    answer_text["answer_text"] = map_text_new_answer(products_cat, opt_answer_text_df, type_filters, filters_def_dict)
    answer_text.drop_duplicates(inplace=True)
    return products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text

if __name__=="__main__":
    products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text = init_df()
    save_obj(products_cat, '../data/products_table')
    save_obj(traffic_cat, '../data/traffic_table')
    save_obj(purchased_cat, '../data/purchased_table')
    save_obj(filters_def_dict, '../data/filters_def_dict')
    save_obj(type_filters, '../data/type_filters')
    save_obj(question_text_df, '../data/question_text_df')
    save_obj(answer_text, '../data/answer_text')

