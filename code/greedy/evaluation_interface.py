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
from PIL import Image, ImageTk


from utils.load_utils import *
from utils.init_dataframes import init_df
import utils.algo_utils as algo_utils
from utils.sampler import sample_answers
from greedy.MaxMI_Algo import max_info_algorithm, opt_step
from utils.build_answers_utils import question_id_to_text, answer_id_to_text, process_answers_filter, answer_text_to_id
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


class Page(Frame):
    def __init__(self, id):
        Frame.__init__(self)
        pages = ["Questions",
                 "Final Product Set"]
        #Label(self, text=pages[id]).pack(fill=BOTH)

class MyApplication(Frame):
    def __init__(self, product_set, traffic_set, purchased_set, question_text_df, answer_text_df, threshold, filters_def_dict, type_filters):

        # Inclusing all necessary data
        self.product_set = product_set
        self.traffic_set = traffic_set
        self.purchased_set = purchased_set
        self.question_text_df = question_text_df
        self.answer_text_df = answer_text_df
        self.threshold = threshold
        self.filters_def_dict = filters_def_dict
        self.type_filters = type_filters

        self.final_question_list = []

        self.question_set = set(algo_utils.get_questions(self.product_set))

        self.root = Tk()
        self.root.title("User Interface - Max_MI algo Test")
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=2)
        self.root.rowconfigure(0, weight=1)

        # Setting up pages (to allow for different screens
        Frame.__init__(self)
        self.page = 0
        self.pages = [Page(x) for x in range(2)]  # creates list of 2 pages
        self.pages[self.page].grid(column=2, row=1, columnspan=3, rowspan=2, sticky=(W, E))


        # Title of the interface - Questions
        self.title = StringVar()
        self.title.set("Question 1")
        self.titleLabel = ttk.Label(self.mainframe, textvariable=self.title, font=("Helvetica", 18))

        # Title of the question
        self.question = IntVar()
        self.question.set(746)
        self.question_text = StringVar()
        self.question_text.set(question_id_to_text(self.question.get(), question_text_df))

        self.questionLabel = ttk.Label(self.mainframe, textvariable=self.question_text).grid(column=2, row=4, columnspan=3, sticky=(W, E))


        # Multiple choice list of answers
        self.answer_set = self.product_set.loc[self.product_set["PropertyDefinitionId"] == int(self.question.get()), "answer"].drop_duplicates().values
        #print(self.answer_text_df["answer_id"])
        #self.answer_set = process_answers_filter(self.question, dict_dict_answers, self.filters_def_dict, self.type_filters) # testing - probably wrong
        print("answer set: {}".format(self.answer_set))
        print("answer set: {}".format(type(self.answer_set)))
        print('int(self.question.get()): {}'.format(int(self.question.get())))
        self.text_answers = build_answers_utils.answer_id_to_text(self.answer_set, int(self.question.get()), self.answer_text_df)
        self.text_answers = np.append(self.text_answers, ['None of the above'])

        print(self.text_answers)

        self.yScroll = Scrollbar(self.mainframe, orient=VERTICAL)  # scroll bar
        self.yScroll.grid(row=6, column=1, sticky=N + S)

        listbox = Listbox(self.mainframe, yscrollcommand=self.yScroll.set, selectmode='multiple')
        for var in self.text_answers:
            listbox.insert(END, var)
        listbox.select_set(0)  # sets the first element
        self.answerList = listbox
        self.answerList.grid(column=2, row=6, columnspan=5, sticky=W)


        # Labels
        self.nb_product_left = len(self.product_set)
        self.nb_question_asked = 1
        self.product_left = StringVar()
        self.product_left.set('Nb products left {}'.format(self.nb_product_left))
        self.question_asked = StringVar()
        self.question_asked.set('Nb question asked {}'.format(self.nb_question_asked))       
        self.productLeftLabel = ttk.Label(self.mainframe, textvariable=self.product_left).grid(column=2, row=16, columnspan=3, sticky=(W, E))
        self.questionAskedLabel = ttk.Label(self.mainframe, textvariable=self.question_asked).grid(column=2, row=17, columnspan=3, sticky=(W, E))

        self.final_products = StringVar()

        
        # Main button Next question
        self.NextButton = ttk.Button(self.mainframe, text="Next", command=self.next).grid(column=7, row=6, sticky=W)



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
        #print(self.answerList.curselection())

        # TODO Acccount for different answer types (value, bin, etc...) - check if needed
        # DONE: add a none option (to differentiate none and idk)
        # TODO Stop simulation when product set is less than threshold
        # TODO Do not print None option if it leads to empty
        # TODO Show answer number if no tex available

        # TODO find better way of dealing with no anwwer than try and except

        # Update answer as answer selected. If no answer given, then consider as 'idk'
        text_values = [self.answerList.get(idx) for idx in self.answerList.curselection()]
        if text_values == ['None of the above']:
            values = ['none']
            print("values: {}".format(text_values))
        elif text_values == []:
            values = ['idk']
            print("values: {}".format(text_values))
        else:
            values = answer_text_to_id(text_values, int(self.question.get()), self.answer_text_df)
        print("values: {}".format(values))

        print("self.question.get(): {}".format(self.question.get())) #For debugging
        # Updating product_set, traffic_set, purchased_set, answer_set and question set
        self.product_set, self.traffic_set, self.purchased_set = algo_utils.select_subset(question=int(self.question.get()), answer=values, product_set=self.product_set, traffic_set=self.traffic_set, purchased_set=self.purchased_set)
        self.question_set = set(algo_utils.get_filters_remaining(self.product_set)) #TODO Question: difference between get_questions and get_filters?

        self.final_question_list.append(int(self.question.get()))
        print("Length Product set: {}".format(len(self.product_set)))#For debugging
        question_set_new = set(algo_utils.get_filters_remaining(self.product_set))
        print("Length Question set new: {}".format(len(question_set_new)))#For debugging
        print("Length Final question list: {}".format(len(self.final_question_list))) #For debugging
        self.question_set = question_set_new.difference(self.final_question_list)
        print("Question set: {}".format(self.question_set))

        # Getting next question from our algo's opt_step # TODO Modify update step for other algorithm
        next_question = opt_step(self.question_set, self.product_set, self.traffic_set, self.purchased_set, use_history=False, df_history=0, alpha=2)
        print("Next question: {}".format(next_question))
        next_question_text = question_id_to_text(next_question, self.question_text_df)

        # Updating number of questions asked
        self.nb_question_asked += 1
        self.question_asked.set('Nb question asked {}'.format(self.nb_question_asked))
        self.title.set("Question {}".format(self.nb_question_asked))

        # Updating number of products left
        self.nb_product_left = len(self.product_set)
        self.product_left.set('Nb products left {}'.format(self.nb_product_left))
        if self.nb_product_left < 1000:
            print("Threshold reached")
            win = Toplevel(self.root)
            win.title('Here is what we can offer you!')
            self.title.set("Your final Product Set")
            self.titleLabel = ttk.Label(win, textvariable=self.title, font=("Helvetica", 18))
            self.final_productsLabel = ttk.Label(win, text=self.product_set['ProductId']).grid(column=2, row=4, columnspan=3, sticky=(W, E))


        # Updating question asked and question set
        #self.question.set("Display new question")
        self.question.set(next_question)
        self.question_text.set(next_question_text)

        # Getting new answer list
        try:
            self.answerList.selection_clear(0, 'end') # clears selected answers IF user selected an answer
            print("Answers cleared")
        except:
            None

        self.answer_set = self.product_set.loc[self.product_set["PropertyDefinitionId"] == int(self.question.get()), "answer"].drop_duplicates().values
        self.text_answers = build_answers_utils.answer_id_to_text(self.answer_set, int(self.question.get()), self.answer_text_df)
        self.text_answers = np.append(self.text_answers, ['None of the above'])

        listbox = Listbox(self.mainframe, yscrollcommand=self.yScroll.set, selectmode='multiple')
        for var in self.text_answers:
            listbox.insert(END, var)
        listbox.select_set(0)  # sets the first element
        self.answerList = listbox
        self.answerList.grid(column=2, row=6, columnspan=5, sticky=W)


        self.answer_set = products_cat.loc[products_cat["PropertyDefinitionId"] == int(self.question.get()), "answer"].drop_duplicates().values.astype(float)

        
        return 1

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':

    from utils.load_utils import load_obj
    from utils.sampler import sample_answers


# TODO change path nanes (somehow had problems with those so had to modify)
    try:
        products_cat = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/products_table')
        traffic_cat = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/traffic_table')
        purchased_cat = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/purchased_table')
        question_text_df = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/question_text_df')
        answer_text_df = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/answer_text')
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

    filters_def_dict = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/filters_def_dict')
    type_filters = load_obj('/Users/Nini/DataScienceLab2018_Digitec/data/type_filters')


    threshold = 10

    print(products_cat["PropertyDefinitionId"].unique())


    app = MyApplication(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text_df, threshold, filters_def_dict, type_filters)
    app.run()

