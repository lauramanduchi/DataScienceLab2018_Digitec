import numpy as np
from load_utils import *
from MaxMI_Algo import max_info_algorithm
from RandomBaseline import random_baseline, get_distinct_products
try:
    products_cat = load_obj('../data/products_table')
    traffic_cat = load_obj('../data/traffic_table')
    purchased_cat = load_obj('../data/purchased_table')
    print("Loaded datsets")
except:
    print("Creating datasets...")
    products_cat, traffic_cat, purchased_cat = init_df()
    
size_test = 25    
y_array = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size_test)
threshold = 50
length_opt = []
length_rdm = []
for y in y_array:
    final_question_list, product_set, y = max_info_algorithm(products_cat, traffic_cat, purchased_cat, threshold, y)
    length_opt.append(len(get_distinct_products(product_set)))
    final_question_list, product_set, y = random_baseline(products_cat, traffic_cat, purchased_cat, threshold, y)
    length_rdm.append(len(get_distinct_products(product_set)))
print('Avg number of questions for optimal {}'.format(np.mean(np.asarray(length_opt))))
print('Std number of questions for optimal {}'.format(np.std(np.asarray(length_opt))))
print('Max number of questions for optimal {}'.format(np.max(np.asarray(length_opt))))
print('Min number of questions for optimal {}'.format(np.min(np.asarray(length_opt))))
print('')
print('Avg number of questions for random {}'.format(np.mean(np.asarray(length_rdm))))
print('Std number of questions for random {}'.format(np.std(np.asarray(length_rdm))))
print('Max number of questions for random {}'.format(np.max(np.asarray(length_rdm))))
print('Min number of questions for random {}'.format(np.min(np.asarray(length_rdm))))