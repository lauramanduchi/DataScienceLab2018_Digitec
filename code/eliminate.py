from load_utils import *
import numpy as np
import algo_utils
import parmap
from build_answers_utils import question_id_to_text, answer_id_to_text


# about parmap
#y = [myfunction(x, argument1, mykeyword=argument2) for x in mylist]
# In parallel:
#y = parmap.map(myfunction, mylist, argument1, mykeyword=argument2)



# TODO LATER ! IDEAS TO IMPROVE
# we can weight the filter by history usage (to avoid having bad filters as first questions)

def expectation_eliminate(question, product_set, traffic_cat): 
    proba_Q = algo_utils.get_proba_Q_distribution_none(question, product_set, traffic_cat)
    possible_answers = proba_Q.index
    proba_Q["eliminate"]=0
    for answer in possible_answers:
        tmp = algo_utils.select_subset(product_set, question=question, answer=np.asarray([answer]))[0]
        proba_Q.loc[answer, "eliminate"] = 1-len(get_distinct_products(tmp))/len(get_distinct_products(product_set)) # 1 - prop_kept = prop_eliminate
    prop = np.sum(proba_Q["final_proba"]*proba_Q["eliminate"])
    return (prop)

# Return question which maximizes eliminate
def opt_step(question_set, product_set, traffic_set, purchased_set):
    MI_matrix = np.zeros([len(question_set), 2])
    mutual_array = np.asarray(parmap.map(expectation_eliminate, \
                                        question_set,\
                                         product_set=product_set,\
                                          traffic_cat=traffic_set, \
                                          pm_pbar=True, \
                                          pm_parallel=True))
    MI_matrix[:,0] = list(question_set)
    MI_matrix[:,1] = mutual_array
    next_question_index = np.argmax(MI_matrix, axis=0)[1]
    next_question = MI_matrix[next_question_index, 0]
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

def max_eliminate_algorithm(product_set, traffic_set, purchased_set, question_text_df, answer_text_df,
                            threshold, y,  answers_y):
    question_set = set(algo_utils.get_questions(product_set))
    final_question_list=[]
    final_question_text_list=[]
    answer_text_list = []
    distinct_products = get_distinct_products(product_set)
    print("There are {} questions we can ask".format(len(question_set)))
    print("There are {} possible products to choose from".format(len(distinct_products)))
    iter = 1
    while not (len(distinct_products) < threshold or len(question_set) == 0):  
        next_question = opt_step(question_set, product_set, traffic_set, purchased_set)
        next_question = int(next_question)
        print("Next question is filter : {}".format(next_question))
        question_text = question_id_to_text(next_question, question_text_df)
        print("Question is: {}".format(question_text))
        final_question_list.append(int(next_question))
        final_question_text_list.append(question_text)
        answer = answers_y.get(next_question)
        answer_text = answer_id_to_text(answer, answer_text_df)
        print("Answer given was: {}".format(answer))
        print("Answer was: {}".format(answer_text))
        answer_text_list.append(answer_text)
        product_set, traffic_set, purchased_set = algo_utils.select_subset(question=next_question, \
                                                                            answer=answer, \
                                                                            product_set=product_set, \
                                                                            traffic_set =traffic_set, \
                                                                            purchased_set = purchased_set)
        question_set = question_set.difference(final_question_list)
        distinct_products = get_distinct_products(product_set)
        print("There are {} possible products to choose from".format(len(get_distinct_products(product_set))))
        iter+=1
    return final_question_list, product_set, y, final_question_text_list, answer_text_list

