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

class MyApplication:
    def __init__(self):
        self.root = Tk()
        self.root.title("Interface prototype one")
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=2)
        self.root.rowconfigure(0, weight=1)

        # Title of the interface
        self.title = StringVar()
        self.title.set("This is the interface title")
        self.titleLabel = ttk.Label(self.mainframe, textvariable=self.title, font=("Helvetica", 18)).grid(column=2, row=1, columnspan=3, rowspan=2, sticky=(W, E))
    
        # Title of the question
        self.question = StringVar()
        self.question.set("This is the fixed question")
        self.questionLabel = ttk.Label(self.mainframe, textvariable=self.question).grid(column=2, row=4, columnspan=3, sticky=(W, E))

        # Multiple choice list of answers
        self.yScroll = Scrollbar(self.mainframe, orient=VERTICAL) #scroll bar
        self.yScroll.grid(row=6, column=1, sticky=N+S)   
        self.answers = StringVar()
        self.answers.set("Ans1 Ans2 Ans3 Ans4 Ans5 Ans6 Ans7 Ans8 Ans9 Ans10 Ans11")
        self.answerList = Listbox(self.mainframe, listvariable= self.answers, yscrollcommand=self.yScroll.set, selectmode='multiple', height=10)
        self.answerList.grid(column=2, row=6, columnspan=3, sticky=W)
    
        # Labels
        self.nb_product_left = 7000
        self.nb_question_asked = 0
        self.product_left = StringVar()
        self.product_left.set('Nb product left {}'.format(self.nb_product_left))
        self.question_asked = StringVar()
        self.question_asked.set('Nb question asked {}'.format(self.nb_question_asked))       
        self.productLeftLabel = ttk.Label(self.mainframe, textvariable=self.product_left).grid(column=3, row=16, columnspan=3, sticky=(W, E))
        self.questionAskedLabel = ttk.Label(self.mainframe, textvariable=self.question_asked).grid(column=3, row=17, columnspan=3, sticky=(W, E))
        
        # Main button Next question
        self.NextButton = ttk.Button(self.mainframe, text="Next", command=self.next).grid(column=6, row=6, sticky=W)



    def next(self):
        """ this is the function called when you press next
        it should TODO
        1. modify the text of the question
        2. modify the list of the answers
        3. update nb product left
        4. update nb question asked

        if underthreshold
        answer list empty (or deleted) 
        question label is "you have finished everything"

        For now it updates the thing just to show its works.
        """
        # prints the index of the selected answers to the console (not the interface)
        # just to remember how to access it for later function
        print(self.answerList.curselection())
        
        # update number question asked done
        self.nb_question_asked += 1
        self.question_asked.set('Nb question asked {}'.format(self.nb_question_asked))
        
        # TODO find the next question
        self.question.set("Display new question")
        
        # TODO find the next answer list
        self.answerList.selection_clear(0, 'end') # clears selected answers
        self.answers.set("NewAns1 NewAns2 NewAns3 NewAns4 NewAns5 NewAns6 NewAns7 NewAns8 NewAns9 NewAns10 NewAns11")
        
        # TODO update number of products left
        self.nb_product_left -= 1
        self.product_left.set('Nb product left {}'.format(self.nb_product_left))
        
        return 1

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = MyApplication()
    app.run()