import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import numpy as np
import pandas as pd

from utils.load_utils import *
from greedy.MaxMI_Algo import max_info_algorithm
from greedy.RandomBaseline import random_baseline
from utils.init_dataframes import init_df
import utils.algo_utils
from utils.algo_utils import get_distinct_products

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
with open(checkpoint_dir +'/lengths.csv', 'w+') as f:
    f.write("random, opt \n")
with open(checkpoint_dir +'/quest.csv', 'w+') as f:
    f.write("random, opt \n")
for y in y_array:
    final_question_list, product_set, y = max_info_algorithm(products_cat, traffic_cat, purchased_cat, threshold, y)
    print('the length of optimal filter was {}'.format(len(get_distinct_products(product_set))))
    length_opt.append(len(get_distinct_products(product_set)))
    opt_quest.append(final_question_list)
    final_question_list, product_set, y = random_baseline(products_cat, traffic_cat, purchased_cat, threshold, y)
    length_rdm.append(len(get_distinct_products(product_set)))
    print('the length of random filter was {}'.format(len(get_distinct_products(product_set))))
    rdm_quest.append(final_question_list)
    with open(checkpoint_dir +'/lengths.csv', 'a+') as f:
        f.write('{}, {} \n'.format(length_opt[-1], length_rdm[-1]))
    with open(checkpoint_dir +'/quest.csv', 'a+') as f:
        f.write('{}, {} \n'.format(opt_quest[-1], rdm_quest[-1]))
"""
res = pd.DataFrame()
res["random"] = length_rdm
res["opt"] = length_opt
res.to_csv(checkpoint_dir +'/lengths.csv', header = True, index = False)

quest = pd.DataFrame()
quest["random"] = rdm_quest
quest["opt"] = opt_quest
quest.to_csv(checkpoint_dir +'/quest.csv', header = True, index = False)
"""
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