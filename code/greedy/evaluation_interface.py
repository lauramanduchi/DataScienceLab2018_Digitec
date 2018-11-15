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
import random as rd


from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from greedy.MaxMI_Algo import max_info_algorithm
from greedy.RandomBaseline import random_baseline
import dagger.dagger_utils as dagger_utils
import utils.build_answers_utils as build_answers_utils
from greedy.evaluation_live import get_next_q_user
# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)

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


threshold = 50
products_set = products_cat.copy()
state = {} # initial state   
done=True 

class Application:
    def __init__(self):
        # init the main window
        self.root = Tk()
        self.root.title("User interface I")

        # init variables
        self.currentQuestion = StringVar()
        self.answer = StringVar()
        self.NumberProductsLeft = IntVar()
        nb_products = algo_utils.get_distinct_products(products_cat)
        self.NumberProductsLeft.set(nb_products) # init value
        
        # init top level frame
        self.ftop = Frame(self.root, width = 300, height= 300, bg='pink')
        self.ftop.pack(fill = BOTH, expand = 1)

        # add score
        self.f1 = Frame(self.ftop, bg='pink')
        self.f1.pack(side=TOP)
        self.f2 = Frame(self.f1, bg='pink')
        self.f2.pack(side=LEFT)
        self.score_title = Label(self.f2, text="Nb Product Left", bg='pink', fg="white", font=("Helvetica", 16))
        self.score_title.grid(row=0, column=1, padx=5)
        self.score = Label(self.f2, textvariable=self.NumberProductsLeft, bg='pink', fg="white", font=("Helvetica", 16) )
        self.score.grid(row=1, column=1, padx=5)
        """
        # add level
        self.score_title = Label(self.f2, text="Level", bg='pink', fg="white", font=("Helvetica", 16))
        self.score_title.grid(row=0,column=2, padx=5)
        self.level = Label(self.f2, textvariable=self.YourLevel, bg='pink',  fg="white", font=("Helvetica", 16))
        self.level.grid(row=1, column=2, padx=5)
        """
        # add the quit button
        self.quit = Button(self.f2, text="QUIT", bg='pink', fg= "white", highlightbackground='pink', \
			padx=3, pady=3, justify='center', bd= 5, command=self.root.destroy, font=("Helvetica", 16))
        self.quit.grid(row=0, column=5, rowspan=2, padx=(150,10))

        
        self.select = Button(self.f2, text="Select", bg='pink', fg= "white", highlightbackground='pink', \
		#	padx=3, pady=3, justify='center', bd= 5, command= #TODO, font=("Helvetica", 16)) #trigger the procedure to get the next thing
        self.question = Label(self.f2, textvariable=self.currentQuestion, bg='pink', fg="white", font=("Helvetica", 16))
        self.answer = #TODO things to display list of possible answer
        #self.can.bind_all('<KeyPress>', self.changeDirection)
        
        # create the game canvas 
        self.can = Canvas(self.ftop, width=300, height= 300)
        self.can.pack(padx=10, pady=10)

        # init the first interface
        self.initQuestions() #TODO

        
