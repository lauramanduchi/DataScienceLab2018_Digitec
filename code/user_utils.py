import pandas as pd
import numpy as np

# Gets the corresponding text equivalent of a questions id
def question_id_to_text(question, question_df):
    try:
        question_text = question_df[question_df["PropertyDefinitionId"] == str(question)]["PropertyDefinition"].values[0]
    except:
        question_text = 'No text equivalent for question'
    return question_text

def answer_id_to_text(answer, answer_df):
    answer_list = []
    for i in answer:
        if i == 'idk':
            answer_list.append('idk')
        elif i == 'none':
            answer_list.append('none')
        else:
            try:
                # answer_list.append(answer_df[answer_df["PropertyDefinitionOptionId"] == str(int(i))].iloc[0][0])
                answer_list.append(answer_df[answer_df["PropertyDefinitionOptionId"] == str(int(i))]["PropertyDefinitionOption"].values[0])
            except:
                answer_list.append('Not found')
    return (answer_list)


'''''''--- TEST ---'''
# if __name__=='__main__':
#     from load_utils import load_obj
#
#     answer_text_df = load_obj('../data/answer_text_df')
#
#     print(type([289. 287. 292.]))
#
#     re.sub("[ ]", " ,", [289. 287. 292.])
#
#     answer = [289. 287. 292.]
#     print(int(answer[0]))
#
#     # print(answer_text_df["PropertyDefinitionId"])
#     # print(answer_text_df[answer_text_df["PropertyDefinitionId"]=='97'])
#     answer_df = answer_text_df
#     print(answer_df[answer_df["PropertyDefinitionOptionId"] == str('idk')]["PropertyDefinitionOption"])