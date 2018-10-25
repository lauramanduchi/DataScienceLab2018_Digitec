'''
# At every answer node, Compute I(X_e,Y;phi_l) and Choose X_e tlq Mututal Information is maximized
'''
from load_utils import *
import random
import numpy as np
import pandas as pd
#from utils import *
from init_dataframes import init_df
import algo_utils
import parmap

# about parmap 
#y = [myfunction(x, argument1, mykeyword=argument2) for x in mylist]
# In parallel:
#y = parmap.map(myfunction, mylist, argument1, mykeyword=argument2)


'''Normalized (# products given answer/total # products given question)'''
def prob_answer(answer, question, product_set, traffic_set):
    p_answer = algo_utils.get_proba_Q_distribution(question, product_set, traffic_set, alpha=1) #Mel function name
    # print(p_answer)
    """
#     with open('../data/probabilities.txt', 'a+') as f:
    with open('../data/probabilities.txt', 'w') as f:
        f.write("p_answer: ")
        f.write("\n")
        p_answer.to_csv(f, header=True)
        f.write("\n")
    """
    return p_answer.loc[float(answer)]["final_proba"]


'''Normalized (# products y bought/total #products bought) + (# products y /total #products)'''
def prob_product(product, product_set, purchased_set):
    p_product = algo_utils.get_proba_Y_distribution(product_set, purchased_set, alpha=1) #Mel function name
    return p_product.loc[product]["final_proba"]

'''
Updates product_set, traffic_set and purchased_set to account for conditionality
'''
def conditional_entropy(answer, question, product_set, traffic_set, purchased_set): #laura purchased_set deleted
    #print('begin conditional entropy')
    product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=answer,
                                                            product_set=product_set, traffic_set =traffic_set,
                                                            purchased_set = purchased_set)
    product_ids = product_set["ProductId"].drop_duplicates().values
    cond_entropy_y = 0
    for product in product_ids:
        prob_y_given_a = prob_product(product, product_set, purchased_set)
        cond_entropy_y += prob_y_given_a * np.log(prob_y_given_a)
    return cond_entropy_y


def mutual_inf(question, product_set, traffic_set, purchased_set):
    #print('begin mutual_info')
    short_mutual_info = 0
    answer_set = product_set.loc[product_set["PropertyDefinitionId"]==question, "answer"].drop_duplicates().values
    # mel 
    proba_anws_array = np.asarray(parmap.map(prob_answer, answer_set, question=question, product_set=product_set, traffic_set=traffic_set))
    #print(proba_anws_array)
    cond_entr_array = np.asarray(parmap.map(conditional_entropy, answer_set, question = question, product_set=product_set, traffic_set=traffic_set, purchased_set=purchased_set))
    #print(cond_entr_array)
    short_mutual_info = -sum(proba_anws_array*cond_entr_array)
    print(short_mutual_info)
    """
    for answer in answer_set:
        short_mutual_info += - prob_answer(answer, question, product_set, traffic_set)* \
                             conditional_entropy(answer, question, product_set, traffic_set, purchased_set)
    """
    return short_mutual_info


# Return question which maximizes MI
def opt_step(question_set, product_set, traffic_set, purchased_set):
    print('begin opt step')
    MI_matrix = np.zeros([len(question_set), 2])
    i = 0
    n = len(question_set)
    for question in question_set:
        print('{} out of {} questions processed'.format(i,n))
        MI_matrix[i] = [question, mutual_inf(question, product_set, traffic_set, purchased_set)]
        i += 1
    next_question = np.argmax(MI_matrix, axis=0)[1]
    return next_question

def get_distinct_products(product_set):
    try:
        distinct_p = product_set.ProductId.unique()
    except AttributeError:
        print("'ProductId' is not a valid column in Product_set, rename it!")
    return distinct_p


'''Our algorith which returns:
 1) sequence of question to ask
 2) final product list
 3) y chosen as input of algo'''
def max_info_algorithm(product_set, traffic_set, purchased_set, threshold, y):
    question_set = set(algo_utils.get_questions(product_set))
    quest_answer_y = algo_utils.get_answers_y(y, product_set)

    final_question_list=[]

    distinct_products = get_distinct_products(product_set)   #laura

    while not (len(distinct_products) < threshold or len(question_set) == 0):     #laura
        next_question = opt_step(question_set, product_set, traffic_set, purchased_set)
        final_question_list.append[next_question]
        answer = quest_answer_y[str(next_question)][0]          #laura
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question, answer=answer, product_set=product_set, traffic_set =traffic_set, purchased_set = purchased_set)
        question_set_new = set(algo_utils.get_filters_remaining(product_set))
        question_set = question_set_new.difference(final_question_list)
        distinct_products = get_distinct_products(product_set) #laura
        print(len(get_distinct_products(product_set)))
    return final_question_list, product_set, y



if __name__=='__main__':
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        print("Loaded datsets")
    except:
        print("Creating datasets...")
        products_cat, traffic_cat, purchased_cat = init_df()
    
    y = products_cat["ProductId"][10]
    threshold = 50
    final_question_list, product_set, y = max_info_algorithm(products_cat, traffic_cat, purchased_cat, threshold, y)
    print("final_question_list: ", final_question_list)
    print("length final product set: ", len(get_distinct_products(product_set)))

    
#Download the data first!
'''
from init_dataframes import init_df
import pandas as pd
products_cat, traffic_cat, purchased_cat = init_df()

products_cat.to_pickle("../data/products_table.pkl")
traffic_cat.to_pickle("../data/traffic_table.pkl")
purchased_cat.to_pickle("../data/purchased_table.pkl")
'''

# products_cat = pd.read_pickle("../data/products_table.pkl")
# traffic_cat = pd.read_pickle("../data/traffic_table.pkl")
# purchased_cat = pd.read_pickle("../data/purchased_table.pkl")
