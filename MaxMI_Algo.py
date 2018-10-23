'''
# At every answer node, Compute I(X_e,Y;phi_l) and Choose X_e tlq Mututal Information is maximized

NOTE: Must get functions from notebook!
'''

import random
import numpy as np
import pandas as pd
from utils import *


'''Normalized (# products given answer/total # products given question)'''
def prob_answer(answer, question, product_set, traffic_set):
    p_answer = get_proba_Q_distribution(question, product_set, traffic_set, alpha=1) #Mel function name
    return p_answer.loc[answer]["final_proba"]


'''Normalized (# products y bought/total #products bought) + (# products y /total #products)'''
def prob_product(product, product_set, purchased_set):
    p_product = get_proba_Y_distribution(product_set, purchased_set, alpha=1) #Mel function name
    return p_product.loc[product]["final_proba"]

'''
Updates product_set, traffic_set and purchased_set to account for conditionality
'''
def conditional_entropy(question, answer, product_set, traffic_set, purchased_set):
    product_set, traffic_set, purchased_set = select_subset(question=question, answer=answer,
                                                            product_set=product_set, traffic_set =traffic_set,
                                                            purchased_set = purchased_set)
    product_ids = product_set["ProductId"].drop_duplicates().values
    cond_entropy_y = 0
    for product in product_ids:
        prob_y_given_a = prob_product(product, product_set, purchased_set)
        cond_entropy_y += prob_y_given_a * np.log(prob_y_given_a)
    return cond_entropy_y


def mutual_inf(question, product_set, traffic_set):
    short_mutual_info = 0
    answer_set = product_set.loc[product_set["PropertyDefinitionId"]==question, "answer"].drop_duplicates().values
    for answer in answer_set:
        short_mutual_info += - prob_answer(answer, question, product_set, traffic_set)* \
                             conditional_entropy(question, answer, product_set, traffic_set)
    return short_mutual_info

# Return question which maximizes MI
def opt_step(question_set, product_set, traffic_set):
    MI_matrix = np.zeros([len(question_set), 2])
    i = 0
    for question in question_set:
        MI_matrix[i] = [question, mutual_inf(question, product_set, traffic_set)]
        i += 1
    next_question = np.argmax(MI_matrix, axis=0)[1]
    return next_question


'''Our algorith which returns:
 1) sequence of question to ask
 2) final product list
 3) y chosen as input of algo'''
def max_info_algorithm(product_set, traffic_set, threshold, y):
    question_set = get_questions(product_set) # TODO
    quest_answer_y = get_answers_y(y, product_set) # TODO

    final_question_list=[]

    while len(product_set) < threshold: #attention number of different products
        next_question = opt_step(question_set, product_set, traffic_set)
        final_question_list.append[next_question]
        answer = quest_answer_y[next_question] # TODO
        product_set, traffic_set, _ = select_subset(question=question, answer=answer, product_set=product_set,
                                                     traffic_set =traffic_set, purchased_set = purchased_set)
        question_set = get_filters_remaining(product_set)

    return final_question_list, product_set, y



'''Testing'''
# Loading question_dic from dataset - downloaded data for category 6 from db
filters_def_dict = load_obj('filters_def_dict')
question_set = list(filters_def_dict.keys())
print(len(question_set))
print("Question set: {}".format(list(question_set)))


# Partial dataset for category 6 - downloaded data for category 6 from db
df = pd.read_csv('category6_select_subset.csv')
random.seed(99)
df = df.iloc[1:100]
purchased_set=None
print(df['ProductId'].value_counts(normalize=False).keys())
print(list(df))

y = 5820741

# Checking that all probabilities are one
total_prob=0
for y in df['ProductId'].value_counts(normalize=False).keys():
    total_prob += prob_product(y, df, purchased_set)
print(total_prob)
print(df['ProductId'].nunique())

# Get question set (from mel's Get filters code)
filters_def_dict.keys()
