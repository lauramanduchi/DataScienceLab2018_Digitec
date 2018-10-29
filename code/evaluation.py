import time
import os
import numpy as np
from load_utils import *
from MaxMI_Algo import max_info_algorithm
from RandomBaseline import random_baseline, get_distinct_products
import pandas as pd
import logging 
from init_dataframes import init_df

try:
    products_cat = load_obj('../data/products_table')
    traffic_cat = load_obj('../data/traffic_table')
    purchased_cat = load_obj('../data/purchased_table')
    print("Loaded datsets")
except:
    print("Creating datasets...")
    products_cat, traffic_cat, purchased_cat = init_df()

time.time()
t = time.strftime('%d%b%y_%H%M%S')
cwd = os.getcwd()
checkpoint_dir = cwd+'/../runs/' + t + '/'
os.makedirs(checkpoint_dir, 0o777)
print('Saving to ' + checkpoint_dir)


size_test = 25    
y_array = np.random.choice(products_cat["ProductId"].drop_duplicates().values, size = size_test)
threshold = 50
length_opt = []
length_rdm = []
opt_quest = []
rdm_quest = []
for y in y_array:
    final_question_list, product_set, y = max_info_algorithm(products_cat, traffic_cat, purchased_cat, threshold, y)
    length_opt.append(len(get_distinct_products(product_set)))
    opt_quest.append(final_question_list)
    final_question_list, product_set, y = random_baseline(products_cat, traffic_cat, purchased_cat, threshold, y)
    length_rdm.append(len(get_distinct_products(product_set)))
    rdm_quest.append(final_question_list)

res = pd.DataFrame()
res["random"] = length_rdm
res["opt"] = length_opt
res.to_csv(checkpoint_dir +'/lengths.csv', header = True, index = False)

quest = pd.DataFrame()
quest["random"] = rdm_quest
quest["opt"] = opt_quest
quest.to_csv(checkpoint_dir +'/quest.csv', header = True, index = False)

with open(checkpoint_dir +'/summary.txt', 'w+') as f:
    f.write('Avg number of questions for optimal {} \n'.format(np.mean(np.asarray(length_opt))))
    f.write('Std number of questions for optimal {} \n'.format(np.std(np.asarray(length_opt))))
    f.write('Max number of questions for optimal {} \n'.format(np.max(np.asarray(length_opt))))
    f.write('Min number of questions for optimal {} \n'.format(np.min(np.asarray(length_opt))))
    f.write('\n')
    f.write('Avg number of questions for random {} \n'.format(np.mean(np.asarray(length_rdm))))
    f.write('Std number of questions for random {} \n'.format(np.std(np.asarray(length_rdm))))
    f.write('Max number of questions for random {} \n'.format(np.max(np.asarray(length_rdm))))
    f.write('Min number of questions for random {} \n'.format(np.min(np.asarray(length_rdm))))
