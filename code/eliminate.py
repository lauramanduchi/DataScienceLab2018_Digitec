from load_utils import *
import numpy as np
import algo_utils
import parmap
from build_answers_utils import question_id_to_text, answer_id_to_text


# about parmap
#y = [myfunction(x, argument1, mykeyword=argument2) for x in mylist]
# In parallel:
#y = parmap.map(myfunction, mylist, argument1, mykeyword=argument2)



# IDEAS TO IMPROVE
# we can weight the filter by history usage (to avoid having bad filters as first questions)
# take our ideas from the first baseline
# check how idk is handled in get proba Q

def expectation_eliminate(question, product_set, traffic_cat): 
    proba_Q = algo_utils.get_proba_Q_distribution_none(question, product_set, traffic_cat)
    possible_answers = proba_Q.index
    proba_Q["eliminate"]=0
    for answer in possible_answers:
        tmp = algo_utils.select_subset(product_set, question=question, answer=np.asarray([answer]))[0]
        proba_Q.loc[answer, "eliminate"] = 1-len(get_distinct_products(tmp))/len(get_distinct_products(product_set))
        # 1 - prop_kept = prop_eliminate
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
    # print(next_question)
    return next_question

def get_distinct_products(product_set):
    try:
        distinct_p = product_set.ProductId.unique()
    except AttributeError:
        print("'ProductId' is not a valid column in Product_set, rename it!")
    return distinct_p



def max_eliminate_algorithm(product_set, traffic_set, purchased_set, question_text_df, answer_text_df,
                            threshold, y,  answers_y):
    '''

    :param threshold: Max number of products we want in final selection
    :param y: Target product
    :param answers_y: All property values of target product
    :return:    1) sequence of question to ask
                2) final product list
                3) y chosen as input of algo
    '''
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


#TODO Remove Threshold
#TODO
def opt_step_random_start(state, product_set, traffic_set, purchased_set, question_text_df, answer_text_df,
                            threshold, y,  answers_y):
    '''
    Modified version of the max_eliminate_algorithm to allow for start at a given state

    :param state: dictionnary of the form ['q1': [a1, a2], 'q2':[b1],  ...]
    :return: next question to be asked
    '''

    # Updating product_set, traffic_set, purchased_set
    for question in state.keys():
        for answer in state[question]:
            product_set, traffic_set, purchased_set = algo_utils.select_subset(question=question,
                                                                       answer=answer,
                                                                       product_set=product_set,
                                                                       traffic_set=traffic_set,
                                                                       purchased_set=purchased_set)
    question_set = set(algo_utils.get_questions(product_set))
    next_question = opt_step(question_set, product_set, traffic_set, purchased_set)
    next_question = int(next_question)
    print("Next question is filter : {}".format(next_question))
    question_text = question_id_to_text(next_question, question_text_df)
    print("Question is: {}".format(question_text))
    return next_question


if __name__ == '__main__':
    import numpy as np

    x = np.asarray([2, 3, 1, 0, 9, 0, 8, 6, 5, 4, 8, 0, 0, 0, 0, 0,1])
    print(x)
    for i in np.argwhere(x > 1):
        print(i)
    print(np.argwhere(x > 1))
    q10a1=12
    q10a2=3
    q3b1=9
    q1e4=80
    q1b4=61
    q1a1=13

    state = {'q10': [q10a1, q10a2], 'q3': [q3b1], 'q1': [q1e4, q1b4, q1a1]}
    print(state)
    for question in state.keys():
        for answer in state[question]:
            print(answer)

