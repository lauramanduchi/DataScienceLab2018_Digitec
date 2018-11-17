import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import os
import numpy as np
import pandas as pd
import argparse
import warnings
from tkinter import *
from tkinter import ttk
import random as rd


from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from greedy.MaxMI_Algo import max_info_algorithm
from greedy.RandomBaseline import random_baseline
import dagger.dagger_utils as dagger_utils
import utils.build_answers_utils as build_answers_utils
#from greedy.evaluation_live import get_next_q_user
# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)
"""
try:
    products_cat = load_obj('../data/products_table')
    traffic_cat = load_obj('../data/traffic_table')
    purchased_cat = load_obj('../data/purchased_table')
    question_text_df = load_obj('../data/question_text_df')
    answer_text_df = load_obj('../data/answer_text')
    print("Loaded datasets")
except:
    print("Creating datasets...")
    products_cat, traffic_cat, purchased_cat, filters_def_dict, type_filters, question_text_df, answer_text = init_df()
    save_obj(products_cat, '../data/products_table')
    save_obj(traffic_cat, '../data/traffic_table')
    save_obj(purchased_cat, '../data/purchased_table')
    save_obj(filters_def_dict, '../data/filters_def_dict')
    save_obj(type_filters, '../data/type_filters')
    save_obj(question_text_df, '../data/question_text_df')
    save_obj(answer_text, '../data/answer_text')
    print("Created datasets")
"""

def next():
    return 1

root = Tk()
root.title("Interface prototype one")

mainframe = ttk.Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


title = StringVar()
title.set("This is the interface title")
question = StringVar()
question.set("This is the fixed question")
nb_product_left = 7000
nb_question_asked = 0
product_left_label = StringVar()
question_asked_label = StringVar()
ttk.Label(mainframe, textvariable=title).grid(column=3, row=1, columnspan=3, sticky=(W, E))
ttk.Label(mainframe, textvariable=question).grid(column=2, row=4, columnspan=3, sticky=(W, E))
ttk.Button(mainframe, text="Next", command=next).grid(column=6, row=6, sticky=W)
OPTIONS = ["Script 1","Script 2","Script 3","Script 4","Script 5"]
l = Listbox(mainframe, selectmode='multiple', height=10)
l.grid(column=2, row=6, columnspan=3, sticky=W)
l.insert('end', OPTIONS)
root.mainloop()