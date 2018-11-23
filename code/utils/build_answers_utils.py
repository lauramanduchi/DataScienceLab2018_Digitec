#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.parser import parse_query_string


def keep_only_useful_URLs(df):
    """ This is a helper function to get the traffic data.
    It removes the lines where the URL is no parsable (as it 
    is useless if the parser returns {}).

    Args:
        df: extract from the traffic data table. Assumes there is a 
            column "RequestUrl" that has to be checked.
    Returns:
        new: input df without the lines where the URL is not parsable.
    """
    new = df.copy()
    for i in df.index.values:
        if i%1000 == 0:
            print(i)
            print(len(new))
        url = new.loc[i, "RequestUrl"]
        # eliminate the row if the parser returns empty dict
        if not bool(parse_query_string(url)):
            new = new.drop(i)
    return(new)

def create_categories(df_category):
    """ Defines the new answers. 
    Note:
        This is a helper function.
    Args:
        df_category: the product table restricted to one single category.

    Returns:
        filters_def_dict: a dict mapping filtersId to new set of possible answers
        type_filters: a dict mapping filters to type of filters (option, bin or value or mixed)
                      {'questionid':'option'|'bin'|'value'|'mixed'|'no_answer'}
    """
    filters_def_dict = {}
    type_filters = {}
    c = 0
    q = 0
    for f in df_category["PropertyDefinitionId"].drop_duplicates().values:
        c+=1
        values_defOpt = df_category.loc[df_category["PropertyDefinitionId"]==f, \
                                    'PropertyDefinitionOptionId'].dropna().drop_duplicates().values
        valuesProp = df_category.loc[df_category["PropertyDefinitionId"]==f, \
                                    'PropertyValue'].dropna().drop_duplicates().values
        
        # Case filter is of 'option' type (i.e. answer is in defined set of possibilities)
        if (len(valuesProp)==0 and len(values_defOpt)>0):
            filters_def_dict.update({str(f): values_defOpt})
            type_filters.update({str(f): 'option'}) #case only optionId
        
        # Case filter is of type 'value' or 'bin' (i.e. answer is a value not an id)
        elif (len(values_defOpt)==0 and len(valuesProp)>0): 
            # Case over than 10 possibles values
            # New answers are 10 bins constructed based on percentiles.
            if len(valuesProp) > 10:
                bins = np.percentile(valuesProp, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                filters_def_dict.update({str(f): bins})
                type_filters.update({str(f): 'bin'})
                q+=1
            # Else keep the original answers
            else:
                filters_def_dict.update({str(f): valuesProp})
                type_filters.update({str(f): 'value'})
        
        # If the answers is sometimes stored as an id and sometimes as a value 
        # in the original dataframe. Keep the original answer.
        elif (len(values_defOpt)>0 and len(valuesProp)>0): # both filled -> put values in optId
            l = set(values_defOpt)
            l2 = set(valuesProp)
            filters_def_dict.update({str(f): np.array(l.union(l2))})
            type_filters.update({str(f): 'mixed'})
        # If there are no answer.
        else:
            print('No answer is provided for filter {}'.format(f))
            type_filters.update({str(f): 'no_answer'})
    print('Have to categorize {} filters out of {}'.format(q,c))
    return(filters_def_dict, type_filters)

def eliminate_filters_no_answers(df, type_filters):
    """To eliminate questions for which there are no 
    value available in the catalog.

    Note:
        First you need to create_categories in order to get the
        type_filters dictonary.

    Args:
        df: input product_catalog to clean
        type_filters: input type_filters dict (cf. create_category)
    
    Returns:
        new: new dataframe with the 'no_answer' filters.
    
    Example:
        >>> df = load_obj(products_cat, '../data/products_table')
        >>> filters_def_dict, type_filters = create_categories(df)
        >>> df = eliminate_filters_no_answers(df, type_filters)
    """
    new = df.copy()
    for f in type_filters:
        if type_filters[f]=='no_answer':
            ind_temp = new.loc[new["PropertyDefinitionId"]==float(f),].index.values
            new = new.drop(ind_temp)
    return(new)

def map_origAnswer_newAnswer(df, filters_def_dict, type_filters):
    """ Function to construct the final 'answer' column.
    Note:
        First run create_category to get filters_def_dcit and 
        type_filters
    
    Args:
        df: input product_catalog with the row answers
        filters_def_dict: as described in create_category
        type_filters: as described in create_category
    
    Returns:
        answers: array of values to be used as the new 'answer' column. 
                Ordered in the same order as the original df index.

    Example: 
        >>> df = load_obj(products_cat, '../data/products_table')
        >>> filters_def_dict, type_filters = create_categories(df)
        >>> df['answer'] = map_origAnswer_newAnswer(df, filters_def_dict, type_filters)
    """
    answers = []
    for i in df.index.values:
        # get current question
        filter = df.loc[i, "PropertyDefinitionId"]
        # construct the new answer depending on the type of question
        if type_filters[str(filter)]=='option':
            answers.append(df.loc[i,"PropertyDefinitionOptionId"])
        elif type_filters[str(filter)]=='value':
            answers.append(df.loc[i,"PropertyValue"])
        # if bin filter map original answer to corresponding bin
        elif type_filters[str(filter)]=='bin':
            bins = filters_def_dict[str(filter)]
            n = len(bins)-1
            j = 0
            while (df.loc[i,"PropertyValue"]>=bins[j] and j<n):
                j=j+1
            answers.append(bins[j-1])
        # if mixed and answer is in id use id otherwise use value
        elif type_filters[str(filter)]=='mixed':
            if np.isnan(df.loc[i,"PropertyDefinitionOptionId"]):
                answers.append(df.loc[i,"PropertyValue"])
            else:
                answers.append(df.loc[i,"PropertyDefinitionOptionId"]) 
    return(answers)

def map_text_new_answer(df, answer_text_df, type_filters, filters_def_dict):
    """ Finds the string corresponding to the new answer.
    Note:
        First construct the new 'answer' column and add it to df.
    
    Args:
        df: input product catalog
        answer_text_df: dataframe with columns question_id, answer_id and answer_text.
        type_filters: as described in create_category
        filters_def_dict: as described in create_category
    
    Returns:
        text_answers: list of text equivalent to each answer ordered as the df dataframe index.
    
    Example:
        >>> df = load_obj(products_cat, '../data/products_table')
        >>> filters_def_dict, type_filters = create_categories(df)
        >>> df['answer'] = map_origAnswer_newAnswer(df, filters_def_dict, type_filters)
        >>> text_answer = map_text_new_answer(df, answer_text_df, type_filters, filters_def_dict)
    """
    text_answers = []
    for i in df.index.values:
        filter = df.loc[i, "PropertyDefinitionId"] 
        if (type_filters[str(filter)]=='option' or type_filters[str(filter)]=='mixed'):
            if len(answer_text_df.loc[answer_text_df["PropertyDefinitionOptionId"] == str(int(df.loc[i, "answer"])),"PropertyDefinitionOption"].values) == 0:
                text_answers.append(str(df.loc[i, "answer"]))
            else:
                text_answers.append(answer_text_df.loc[answer_text_df["PropertyDefinitionOptionId"] == str(int(df.loc[i, "answer"])),"PropertyDefinitionOption"].values[0])
                #print(answer_text_df.loc[answer_text_df["PropertyDefinitionOptionId"] == str(int(df.loc[i, "answer"])),"PropertyDefinitionOption"].values[0])
        elif type_filters[str(filter)]=='bin':
            bins = filters_def_dict[str(filter)]
            idx = np.where(bins == df.loc[i, "answer"])[0]
            try:
                my_string = '[{}-{}]'.format(bins[idx], bins[idx+1])
            except(IndexError):
                my_string = 'over {}'.format(bins[idx])
            text_answers.append(my_string)
        else:
            text_answers.append(str(df.loc[i, "answer"]))
    return(text_answers)

def filters_answers_per_requestURL(RequestUrl):
    """ Gets the filters and corresponding answers used in one RequestUrl
    Note:
        Uses the parser given by Digitec.
    Args:
        RequestURL: substring of URL as in RequestUrl column of traffic table
    Returns:
        filters: array of question id
        dict_dict_answers: array of dict of raw answers, {filters_id: [{dict_answers}]}.
                            ex: {'7302': [{'PropertyDefinitionOptionIds': ['5767']}]}
    """
    result = parse_query_string(RequestUrl) 
    filters = []
    dict_dict_answers = {}
    # case it is not parsable
    if not bool(result): 
        return(filters, dict_dict_answers)
    try:
        d = result["PropertyGroup"]
    except:
        # case property group not found
        return(filters, dict_dict_answers)
    for _, group_dict in d.items():
        propdef_dict = group_dict['PropertyDefinition']
        for propdef, optProp in propdef_dict.items():
            filters.append(propdef) # PropertyDefinitionId
            temp = []
            temp.append(optProp)
            dict_dict_answers.update({propdef: temp})
    return(filters, dict_dict_answers) 

def hist_to_new(filtername, filters_def_dict, type_filters, min_value=None, max_value=None):
    """ Finds the right bin or value from one historic answer i.e. map historic user input to 
    our new answer category. Also handles the case where the filter used on the website was of type
    min / max. Helper function for process_answers_filter.
    
    Note:
        If min>max or biggest possible returns []
        If max<smallest possible []

    Args:
        filtername: questionId
        filters_def_dict: as described in create_category
        type_filters: as described in create_category
        min_value: min value selected in case of min/max selection
        max_value: max value selected in case of min/max selection

    Returns:
        the new answer
    """
    filtername = str(filtername)
    try:
        if max_value==None:
            if (type_filters[filtername]=="value" or type_filters[filtername]=="mixed"):
                return([min_value]) # nothing has to be done
            elif (type_filters[filtername]=="bin"):
                bins = filters_def_dict[filtername]
                n= len(bins)-1
                j=0
                while (float(min_value)>=bins[j] and j<n):
                    j=j+1
                return([bins[j-1]]) # find the right bin corresponding to the chosen value
        elif min_value > max_value:
            return(np.nan)
        else:
            bins = filters_def_dict[filtername]
            n = len(bins)
            if min_value > bins[n-1]:
                return(np.nan)
            elif max_value < bins[0]:
                return(np.nan)
            else:
                i = 1
                while (min_value >= bins[i] and i<(n-1)):
                    i+=1
                j = i-1
                while (j<n and max_value >= bins[j]):
                    j+=1
                return(bins[(i-1):j]) #find the bins corresponding to the chosen range
    except KeyError:
        print('The filter {} is not in the current product database'.format(filtername))
        return(np.nan)


def process_answers_filter(filtername, total_answer_dict, filters_def_dict, type_filters):
    """ Transforms the answers dict (extracted from original traffic data by the 
    filters_answers_per_requestURL function) to one array of new answers 
    (as defined in create_category function) for one filter.
   
    Args:
       filtername: questionId
       total_answer_dict: one dict returned by filters_answers_per_requestURL
                         in {'PropertyDefinitionOptionIds': ['5767']}
       filters_def_dict: as described in create_category
       type_filters: as described in create_category

    Returns:
        answers: array of new answers
    
    Example:
        >>> RequestUrl = /de/s1/producttype/notebook-6?opt=f168334|v68-1:0.39624|v68-1:0.3556|v68-1:0.35306|v1649-11297:128000000000|v1649-11297:180000000000&bra=5&pdo=13-7302:5767&tagIds=614
        >>> filters, dict_dict_answers = filters_answers_per_requestURL(RequestUrl)
        >>> print(filters)
            ['7302']
        >>> print(dict_dict_answers)
            {'7302': [{'PropertyDefinitionOptionIds': ['5767']}]}
        >>> process_answers_filter(7302, {'PropertyDefinitionOptionIds': ['5767']}, filters_def_dict, type_filters)
            ['5767']
    """
    answers = []
    answer_dict = total_answer_dict[str(filtername)]
    for answers_item in answer_dict:
        to_categorize = False
        if isinstance(answers_item, dict): 
            for answerType, value in answers_item.items():
                if answerType == 'PropertyDefinitionOptionIds':
                    answers.extend(value)
                if answerType == 'Max':
                    max_value = value
                    to_categorize = True
                if answerType == 'Min':
                    min_value = value
                    to_categorize = True
            if to_categorize:
                answers.extend(hist_to_new(filtername, filters_def_dict, type_filters, min_value, max_value))
        else:
            # case where we have to map old answer to new answer.
            # also handling the min/max selection
            answers.extend(hist_to_new(filtername, filters_def_dict, type_filters, answers_item))
    return(list(set(answers)))

def process_all_traffic_answers(traffic_df, purchased_cat, filters_def_dict, type_filters):
    """ Wrapper function to process all answers from the dataset. And performs inner join with purchased_cat
    to have the item_bought_id in the traffic dataset. 

    Note:
        Uses process_answers_filter to map to new answers
        Assuming you have eliminated all invalid urls from extracted traffic data 
        with keep_only_useful_URLs
    
    Args:
       traffic_df: traffic table as extracted in init_dataframe()
       purchased_cat: table of purchased articles as extracted in init_dataframe()
       filters_def_dict: as described in create_category
       type_filters: as described in create_category
    
    Returns:
        res: processed traffic data with colums "SessionId", "answers_selected", "Items_ProductId"
    """
    # assuming you have eliminated all invalid urls
    idx = traffic_df.index.values
    urlsList = traffic_df["RequestUrl"].values
    for i, url in zip(idx, urlsList):
        ans_d = {}
        try:
            filters, dict_dict_answers = filters_answers_per_requestURL(url)
            for f in filters: 
                tmp_new_ans = process_answers_filter(f, dict_dict_answers, filters_def_dict, type_filters)
                ans_d.update({f: tmp_new_ans})
            if bool(ans_d):
                traffic_df.loc[i, "answer"] = [ans_d]
            else:
                traffic_df.loc[i, "answer"] = [dict()]
        except TypeError:
            traffic_df.loc[i, "answer"] =  [dict()]
    session_array = traffic_df["SessionId"].drop_duplicates().values
    res = pd.DataFrame()
    res["SessionId"] = session_array
    res.index = session_array
    for s in session_array:
        a_array = traffic_df.loc[traffic_df["SessionId"]==s, "answer"].values
        temp = {}
        for answer_dict in a_array:
            for key, value in answer_dict.items():
                if key in temp:
                    temp[key] = list(set(temp[key]).union(set(value)))
                else:
                    temp.update({key: value})
        if (temp == {}):
            res = res.drop(s)
        else:
            res.loc[res["SessionId"]==s, "answers_selected"] = [temp]
    res = res.merge(purchased_cat, how='inner', left_on="SessionId", right_on="SessionId")[["SessionId","answers_selected", "Items_ProductId"]]
    return(res)


def question_id_to_text(question, question_df):
    try:
        question_text = question_df.loc[question_df["PropertyDefinitionId"] == str(int(question)), "PropertyDefinition"].values[0]
    except IndexError:
        question_text = 'No text equivalent for question'
    return question_text

def question_text_to_id(question_text, question_df):
    try:
        question_text = question_df.loc[question_df["PropertyDefinition"] == str(question_text), "PropertyDefinitionId"].values[0]
    except IndexError:
        question_text = 'No id equivalent for question'
    return question_text

def answer_id_to_text(answer, question, answer_df):
    answer_list = []
    for i in answer:
        if i == 'idk':
            answer_list.append('idk')
        elif i == 'none':
            answer_list.append('none')
        else:
            try:
                answer_list.append(answer_df.loc[(answer_df["answer_id"] == i) & (answer_df["question_id"] == int(question)), "answer_text"].astype(str).values[0])
            except TypeError:
                answer_list.append(i)
            except IndexError:
                answer_list.append('Not Found: ' + str(i))
    return (answer_list)

def answer_text_to_id(answer_text, question, answer_df):
    answer_list = []
    for i in answer_text:
        print(i)
        if i == 'idk':
            answer_list.append('idk')
        elif i == 'none':
            answer_list.append('none')
        else:
            try:
                answer_list.append(answer_df.loc[(answer_df["answer_text"] == str(i)) & (answer_df["question_id"] == int(question)), "answer_id"].astype(float).values[0])
            #except TypeError:
            #    answer_list.append(i)
            except IndexError:
                print(i)
                if i.startswith("Not Found"):
                    print(str(i.split(": ")[1]))
                    answer_list.append(str(i.split(": ")[1]))
                else:
                    answer_list.append(i)

    return (answer_list)



# To test some functions
if __name__=='__main__':
    from utils.load_utils import load_obj
    from utils.sampler import sample_answers
    try:
        products_cat = load_obj('../data/products_table')
        traffic_cat = load_obj('../data/traffic_table')
        purchased_cat = load_obj('../data/purchased_table')
        question_text_df = load_obj('../data/question_text_df')
        answer_text = load_obj('../data/answer_text')
        print("Loaded datsets")
    except:
        print("Creating datasets...")
        #products_cat, traffic_cat, purchased_cat = init_df()

    y = products_cat["ProductId"][10]
    threshold = 50
    print(products_cat["ProductId"].dtype) #int
    print(products_cat["PropertyDefinitionId"].dtype) #int
    print(products_cat["answer"].dtype) #float
    answers_y = sample_answers(y, products_cat)
    for key, answer in answers_y.items():
        print(answer_id_to_text(answer, key, answer_text))
    #final_question_list, product_set, y = random_baseline(products_cat, traffic_cat, purchased_cat, threshold, y)
    #print("final_question_list: ", final_question_list)
   # print("length final product set: ", len(get_distinct_products(product_set)))