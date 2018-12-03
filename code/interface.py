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
from utils.build_answers_utils import question_id_to_text, answer_id_to_text, process_answers_filter
from greedy.RandomBaseline import random_baseline
import dagger.dagger_utils as dagger_utils
import utils.build_answers_utils as build_answers_utils
from dagger.model import create_model

#from greedy.evaluation_live import get_next_q_user
# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)


class MyApplication(Frame):
    def __init__(self, product_set, traffic_set, purchased_set, question_text_df, answer_text_df, threshold, filters_def_dict, type_filters):
        self.use='dagger'
        #load the NN if necessary
        if self.use=='dagger':
            run = 'default' #TODO how to customize this easliy?
            model_dir = '../training_dagger/{}'.format(run)
            checkpoint_model = model_dir+'/cp.ckpt'
            print('Loading the latest model from {}'.format(checkpoint_model))
            self.length_state = len(dagger_utils.get_onehot_state({}, filters_def_dict))
            self.number_filters = len(filters_def_dict.keys())
            self.model = create_model(self.number_filters, self.length_state, h1=128, h2=122)
            self.model.load_weights(checkpoint_model)
            self.state = {}

        # Including all necessary data
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


        # Title of the interface - Questions
        self.title = StringVar()
        self.title.set("Question 1")
        self.titleLabel = ttk.Label(self.mainframe, textvariable=self.title, font=("Helvetica", 18)).grid(column=2, row=1, columnspan=3, sticky=(W, E))

        # Title of the question
        self.question = IntVar()
        #get first question
        if self.use=='maxMI':
            self.next_question = opt_step(self.question_set, 
                                     self.product_set, 
                                     self.traffic_set, 
                                     self.purchased_set, 
                                     a_hist=1, 
                                     df_history=df_history)
        else:
            self.next_question = dagger_utils.dagger_one_step(self.model, self.state, self.number_filters, self.filters_def_dict)

        self.question.set(self.next_question)
        self.question_text = StringVar()
        self.question_text.set(question_id_to_text(self.question.get(), question_text_df))
        self.questionLabel = ttk.Label(self.mainframe, textvariable=self.question_text).grid(column=2, row=4, columnspan=3, sticky=(W, E))


        # Multiple choice list of answers
        self.answer_set = self.product_set.loc[self.product_set["PropertyDefinitionId"] == int(self.question.get()), "answer"].drop_duplicates().values
        print("answer set: {}".format(self.answer_set))
        print("answer set: {}".format(type(self.answer_set)))
        print('int(self.question.get()): {}'.format(int(self.question.get())))
        self.text_answers = build_answers_utils.answer_id_to_text(self.answer_set, int(self.question.get()), self.answer_text_df)


        self.yScroll = Scrollbar(self.mainframe, orient=VERTICAL)  # scroll bar
        self.yScroll.grid(row=6, column=1, sticky=N + S)

        listbox = Listbox(self.mainframe, yscrollcommand=self.yScroll.set, selectmode='multiple')
        for var in self.text_answers:
            listbox.insert(END, var)
        listbox.select_set(0)  # sets the first element
        self.answerList = listbox
        self.answerList.grid(column=2, row=6, columnspan=5, sticky=W)


        # Labels
        self.nb_product_left = len(self.product_set["ProductId"].unique())
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
        # Update answer as answer selected. If no answer given, then consider as 'idk'
        id_values = [self.answer_set[idx] for idx in self.answerList.curselection()]
        if id_values == []:
            values = ['idk']
            print("values: {}".format(id_values))
        else:
            values = id_values
        print(self.answer_set)
        print(self.answerList.curselection())
        print("values: {}".format(values))
        self.state[self.next_question] = list(values)
        print(self.state)
        print("self.question.get(): {}".format(self.question.get())) #For debugging
        # Updating product_set, traffic_set, purchased_set, answer_set and question set
        self.product_set, self.traffic_set, self.purchased_set = algo_utils.select_subset(question=self.question.get(), answer=values, product_set=self.product_set, traffic_set=self.traffic_set, purchased_set=self.purchased_set)
        self.question_set = set(algo_utils.get_questions(self.product_set))

        self.final_question_list.append(int(self.question.get()))
        print("Length Product set: {}".format(len(self.product_set)))#For debugging
        question_set_new = set(algo_utils.get_questions(self.product_set))
        print("Length Question set new: {}".format(len(question_set_new)))#For debugging
        print("Length Final question list: {}".format(len(self.final_question_list))) #For debugging
        self.question_set = question_set_new.difference(self.final_question_list)
        print("Question set: {}".format(self.question_set))

        # Getting next question from our algo's opt_step # TODO Modify update step for other algorithm
        if self.use=='maxMI':
            self.next_question = opt_step(self.question_set, 
                                     self.product_set, 
                                     self.traffic_set, 
                                     self.purchased_set, 
                                     a_hist=1, 
                                     df_history=df_history)
        else:
            self.next_question = dagger_utils.dagger_one_step(self.model, self.state, 
                                                         self.number_filters, self.filters_def_dict)
        
        print("Next question: {}".format(self.next_question))
        next_question_text = question_id_to_text(self.next_question, self.question_text_df)

        # Updating number of questions asked
        self.nb_question_asked += 1
        self.question_asked.set('Nb question asked {}'.format(self.nb_question_asked))
        self.title.set("Question {}".format(self.nb_question_asked))

        # Updating number of products left
        self.nb_product_left = len(self.product_set["ProductId"].unique())
        self.product_left.set('Nb products left {}'.format(self.nb_product_left))

        # Updating question asked and question set
        self.question.set(self.next_question)
        self.question_text.set(next_question_text)

        # Getting the answers
        self.answer_set = self.product_set.loc[self.product_set["PropertyDefinitionId"] == int(self.question.get()), "answer"].drop_duplicates().values
        self.text_answers = build_answers_utils.answer_id_to_text(self.answer_set, int(self.question.get()), self.answer_text_df)
        
        # If number of products lower than threshold, display final set of products
        print("Number products left: {}".format(len(self.product_set["ProductId"].drop_duplicates())))
        if (len(self.product_set["ProductId"].unique()) < self.threshold) or (len(self.text_answers)==1):
            print("Threshold reached")
            win = Toplevel(self.root)
            win.title('Here is what we can offer you!')
            self.title.set("----   Your final Product Set   ----")
            self.titleLabel = ttk.Label(win, textvariable=self.title, font=("Helvetica", 18)).grid(column=2, row=1, columnspan=10,sticky=(W, E))
            self.final_productsLabel = ttk.Label(win, text="\n".join(map(str, self.product_set['ProductId'].unique()))) \
                                                .grid(column=2, row=4,columnspan=3, sticky=(W, E))
            self.quit()
            return 1

        # Getting new answer list
        try:
            self.answerList.selection_clear(0, 'end') # clears selected answers IF user selected an answer
            print("Answers cleared")
        except:
            None

        listbox = Listbox(self.mainframe, yscrollcommand=self.yScroll.set, selectmode='multiple')
        for var in self.text_answers:
            listbox.insert(END, var)
        listbox.select_set(0)  # sets the first element
        self.answerList = listbox
        self.answerList.grid(column=2, row=6, columnspan=5, sticky=W)
        self.answer_set = self.product_set.loc[self.product_set["PropertyDefinitionId"] == int(float(self.question.get())), "answer"].drop_duplicates().values.astype(float)

    def run(self):
        self.root.mainloop()

    def quit(self):
        self.mainframe.destroy()

if __name__ == '__main__':

    from utils.load_utils import load_obj
    from utils.sampler import sample_answers


    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        question_text_df = load_obj('../data/question_text_df')
        answer_text_df = load_obj('../data/answer_text')
        filters_def_dict = load_obj('../data/filters_def_dict')
        type_filters = load_obj('../data/type_filters')
    except:
        print("Missing datasets...")

    try:
        df_history = load_obj('../data/df_history')
    except:
        df_history = algo_utils.create_history(traffic_cat, question_text_df)
        save_obj(df_history, '../data/df_history')
        print("Created history")
        print("Loaded datasets")
    threshold = 50

    print(products_cat["PropertyDefinitionId"].unique())


    app = MyApplication(products_cat, traffic_cat, purchased_cat, question_text_df, answer_text_df, threshold, filters_def_dict, type_filters)
    app.run()