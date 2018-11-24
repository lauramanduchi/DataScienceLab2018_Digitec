#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is used to extract data from the Digitec database and to preprocess it.
It saves the needed dataframes.
"""

from sqlalchemy import create_engine,text
import pandas as pd
import numpy as np
import time

from utils.parser import parse_query_string
from utils.build_answers_utils import keep_only_useful_URLs, create_categories, \
                                      eliminate_filters_no_answers, map_origAnswer_newAnswer, \
                                      process_all_traffic_answers, map_text_new_answer
from utils.load_utils import load_obj, save_obj, batch


def init_df():
    """ This function loads the data from the SQL server
    and preprocesses it according to our needs.

    Note:
        This code can only be run of Digitec's machine.

    Returns:
        products_cat: extract of product catalog for category 6
        purchased_cat: purchases from products of category 6.
                       only keep purchases where one unique productId was bought.
        traffic_cat: table containing the filters used for purchases in purchased_cat.
        filters_def_dict: dict where key is questionId
                            value is array of all possible (modified) answers
        type_filters: dict {questionId: 'mixed'|'bin'|'value'|'option'}
        question_text_df: dataframe with columns PropertyDefinitionId
                          and PropertyDefinition (string of question)
        answer_text: dataframe with columns question_id, answer_id and answer_text.
    """
    # =============== DATABASE SETUP =========== #
    # Initialize connection to the machine
    engine = create_engine('postgresql://dslab:dslab2018@localhost/dslab')
    c = engine.connect()

    # Number of the category to use
    cat = 6
    
    #================ DATA EXTRACTION =========== #
    # Reduced purchased
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

    # Extract relevant products_cat
    t1 = time.time()
    products_cat = pd.read_sql_query('''
    SELECT "ProductId", "BrandId", "ProductTypeId", "PropertyValue", "PropertyDefinitionId", "PropertyDefinitionOptionId"
    FROM product_only_ids
    WHERE "ProductTypeId"='{}' ;
    '''.format(cat), c)
    t2 = time.time()
    print('Created product_cat in {}s.'.format(t2-t1))
    print('Found {} items'.format(len(products_cat['ProductId'].drop_duplicates().values)))

    # Products purchased from selected category
    t1 = time.time()
    productIdsCat = pd.read_sql_query('''
    SELECT DISTINCT "ProductId"
    FROM product_only_ids
    WHERE "ProductTypeId"='{}' ;
    '''.format(cat), c)
    t2 = time.time()
    print('Created productIdsCat in {}s.'.format(t2-t1))

    t1 = time.time()
    purchased_cat = pd.merge(productIdsCat, reduced_purchased, \
                    left_on="ProductId", right_on="Items_ProductId", \
                    how = "inner")
    t2 = time.time()
    print('Created purchased_Cat in {}s.'.format(t2-t1))
    print('Found {} sold items. And {} unique product id'.format(len(purchased_cat), len(purchased_cat["ProductId"])))

    # Extract of relevant traffic data (batch if necessary)
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
        # Only extract traffic data that can be parsed
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


    # ================= DATA PREPROCESSING =============== #

    # Adding Brands (in a separate column) as filter in products_cat
    print("Adding brands as property in dataframe...")
    i = 0
    brandId = 99999
    for p in products_cat["ProductId"].drop_duplicates():
        brand = products_cat.loc[products_cat['ProductId'] == p]["BrandId"].drop_duplicates()
        brand = brand.values[0]
        producttype_id = products_cat.loc[products_cat['ProductId'] == p]["ProductTypeId"].drop_duplicates().values[0]
        newrow = pd.Series([p, brand, producttype_id, brandId, brand],
                           index=["ProductId", "BrandId", "ProductTypeId", "PropertyDefinitionId",
                                  "PropertyDefinitionOptionId"],
                           name=str(int(i + len(products_cat) + 1)))
        products_cat = products_cat.append(newrow)
        i += 1
    print("Added brands as property")


    # New answer definition
    filters_def_dict, type_filters  = create_categories(products_cat)
    products_cat = eliminate_filters_no_answers(products_cat, type_filters)
    products_cat["answer"] = map_origAnswer_newAnswer(products_cat, filters_def_dict, type_filters)

    # Map traffic data to new answers
    traffic_cat = process_all_traffic_answers(traffic_cat, purchased_cat, filters_def_dict, type_filters)

    # Get text of the questions
    question_text_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinition", "PropertyDefinitionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)
    print('Done question-to-text dataframe')

    # Adding brand to question_text
    newrow_question_text = pd.Series([str("Brand"), brandId],
                                     index=["PropertyDefinition", "PropertyDefinitionId"],
                                     name=str(len(question_text_df)))
    question_text_df = question_text_df.append(newrow_question_text)

    # Get original text of answers
    opt_answer_text_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinitionOption", "PropertyDefinitionOptionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)

    # Process the original to new answers if necessary
    print('Begin answer-to-text dataframe')
    answer_text = pd.DataFrame()
    answer_text["answer_id"] = products_cat["answer"]
    answer_text["question_id"] = products_cat["PropertyDefinitionId"]
    answer_text["answer_text"] = map_text_new_answer(products_cat, opt_answer_text_df, type_filters, filters_def_dict)

    # Adding brands to answer_text
    brand_text_df = pd.read_csv("../data/brands.csv")
    brand_text_df['question_id'] = brandId
    brand_text_df.columns = ["answer_id", "answer_text", "question_id"]  # Renaming columns (so same as answer_text)
    answer_text = answer_text.append(brand_text_df)  # Appending brands dataframe to answer_text
    answer_text.drop_duplicates(inplace=True)

    return products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text

if __name__=="__main__":
    """ 
    Call the init function and save all the objects.
    """
    products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text = init_df()
    save_obj(products_cat, '../data/products_table')
    save_obj(traffic_cat, '../data/traffic_table')
    save_obj(purchased_cat, '../data/purchased_table')
    save_obj(filters_def_dict, '../data/filters_def_dict')
    save_obj(type_filters, '../data/type_filters')
    save_obj(question_text_df, '../data/question_text_df')
    save_obj(answer_text, '../data/answer_text')