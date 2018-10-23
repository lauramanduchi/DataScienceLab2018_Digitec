'''
# At every answer node, Compute I(X_e,Y;phi_l) and Choose X_e tlq Mututal Information is maximized

NOTE: Must get functions from notebook!
'''

import random
import numpy as np
import pandas as pd
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import matplotlib
import matplotlib.pyplot as plt
from prototype1 import *
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

product_set = df.iloc[1:]
print("Product set: {}".format(product_set))
print("Test1: {}".format(product_set['ProductId']))
print("Test2: {}".format(product_set['ProductId'].value_counts(normalize=True).get(6337059)))
# for y in product_set:
#     print("y in product set: {}".format(y))
# print("MI: {}".format(mutual_inf(question_set, product_set)))
MI_matrix = mutual_info_matrix(question_set, product_set)
print("MI matrix: {}".format(MI_matrix))
print("argmax MI: {}".format(np.argmax(MI_matrix, axis=0)[1]))
print("next question: {}".format(MI_matrix[np.argmax(MI_matrix, axis=0)[1]][0]))
next_question_index = MI_matrix[np.argmax(MI_matrix, axis=0)[1]][0]
next_question= MI_matrix[next_question_index][0]
print(next_question)
print("new question set: {}".format(question_set.remove(next_question)))
print("question_set: {}".format(len(question_set)))


print(question_set[1:3])
filters = question_set[1:3]
G = createGraph(filters)
G = sub_tree_from_currentnode(G, 0)
plt.title('Question Tree')
pos = graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=True, arrows=True)
print(G.nodes())

print([x for x in G.nodes() if x!=0 and len(x)==2])
print([x[-1:] for x in G.nodes() if x!=0 and len(x)==4])

print(min([x for x in G.nodes() if x!=0], key=len))
print(len(min([x for x in G.nodes() if x!=0], key=len)))
print(type(max([x for x in G.nodes() if x!=0])))
print('ADCB'[-1:])
print(len(max([x for x in G.nodes() if x!=0], key=len)))
# print(len(list(G.successors('Q1'))))
# for next_question in (list(G.successors('Q1'))):
#     print(next_question)

plt.show()


