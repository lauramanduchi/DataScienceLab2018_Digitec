#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from parser import parse_query_string

def create_categories(df_category):
    """ Defines the new answers. 
    Args:
        df_category: the product table restricted to one single category.
    
    Returns:
        result: a dict mapping filters to set of possible answers
        type_filters: a dict mapping filters to type of filters (option, bin or value or mixed)
    """
    result = {}
    type_filters = {}
    c = 0
    q = 0
    for f in df_category["PropertyDefinitionId"].drop_duplicates().values:
        c+=1
        values_defOpt = df_category.loc[df_category["PropertyDefinitionId"]==f, \
                                    'PropertyDefinitionOptionId'].dropna().drop_duplicates().values
        valuesProp = df_category.loc[df_category["PropertyDefinitionId"]==f, \
                                    'PropertyValue'].dropna().drop_duplicates().values
        if (len(valuesProp)==0 and len(values_defOpt)>0):
            result.update({str(f): values_defOpt})
            type_filters.update({str(f): 'option'}) #case only optionId
        elif (len(values_defOpt)==0 and len(valuesProp)>0): 
            # case only value ids
            if len(valuesProp) > 10:
                _, bins = np.histogram(valuesProp)
                result.update({str(f): bins})
                type_filters.update({str(f): 'bin'})
                q+=1
            else:
                result.update({str(f): valuesProp})
                type_filters.update({str(f): 'value'})
        elif (len(values_defOpt)>0 and len(valuesProp)>0): # both filled -> put values in optId
            l = set(values_defOpt)
            l2 = set(valuesProp)
            result.update({str(f): np.array(l.union(l2))})
            type_filters.update({str(f): 'mixed'})
        else:
            print('No answer is provided for filter {}'.format(f))
            type_filters.update({str(f): 'no_answer'})
    print('Have to categorize {} filters out of {}'.format(q,c))
    return(result, type_filters)

def eliminate_filters_no_answers(df, type_filters):
    new = df.copy()
    for f in type_filters:
        if type_filters[f]=='no_answer':
            ind_temp = new.loc[new["PropertyDefinitionId"]==float(f),].index.values
            new = new.drop(ind_temp)
    return(new)

def map_origAnswer_newAnswer(df, filters_def_dict, type_filters):
    """ finds the new answer for each row of the dataframe, returns list of new values.
    """
    answers = []
    print(len(df.index.values))
    for i in df.index.values:
        filter = df.loc[i, "PropertyDefinitionId"]
        if type_filters[str(filter)]=='option':
            answers.append(df.loc[i,"PropertyDefinitionOptionId"])
        elif type_filters[str(filter)]=='value':
            answers.append(df.loc[i,"PropertyValue"])
        elif type_filters[str(filter)]=='bin':
            bins = filters_def_dict[str(filter)]
            n = len(bins)-1
            j = 0
            while (df.loc[i,"PropertyValue"]>=bins[j] and j<n):
                j=j+1
            answers.append(bins[j-1])
        elif type_filters[str(filter)]=='mixed':
            if np.isnan(df.loc[i,"PropertyDefinitionOptionId"]):
                answers.append(df.loc[i,"PropertyValue"])
            else:
                answers.append(df.loc[i,"PropertyDefinitionOptionId"]) 
    return(answers)

def filters_answers_per_requestURL(RequestUrl):
    """
    Returns for given RequestUr:
    filters: [filters]
    dict_dict_answers: array of dict of raw answers, {filters_id: [{dict_answers}]}
    propgroup_list: list of filter group (maybe not needed) [property group] 
    """
    result = parse_query_string(RequestUrl) 
    propgroup_list = [] # optional
    filters = []
    dict_dict_answers = {}
    if not bool(result): # case it is not parsable
        return(propgroup_list, filters, dict_dict_answers)
    try:
        d = result["PropertyGroup"]
    except:
        # print('no PropertyGroup')
        return(propgroup_list, filters, dict_dict_answers)
    for propgroup, group_dict in d.items():
        propgroup_list.append(propgroup)
        propdef_dict = group_dict['PropertyDefinition']
        for propdef, optProp in propdef_dict.items():
            filters.append(propdef) # PropertyDefinitionId
            temp = []
            temp.append(optProp)
            dict_dict_answers.update({propdef: temp})
    return(propgroup_list, filters, dict_dict_answers) 

def categorize(filtername, filters_def_dict, type_filters, min_value, max_value=None):
    """
    finds the right bins or value i.e. map historic user input to 
    our categorized answers
    if min>max or biggest possible returns []
    if max<smallest possible []
    TEST OK
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
            list_bins = []
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
        print('the filter is not in the current product database')
        return(np.nan)


def process_answers_filter(filtername, total_answer_dict, filters_def_dict, type_filters):
    """
    Transforms the answers dict to a array of answers from the new answer column. To map
    to our new system of asnswer.
    Min - Max - Value - OptionsIds is tested.
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
            if to_categorize:
                answers.extend(categorize(filtername, filters_def_dict, type_filters, min_value, max_value))
        else:
            # case where we have directly the answer
            # no PropertyDefinitionsOptionsIDs but PropertyValue I guess
            answers.extend(categorize(filtername, filters_def_dict, type_filters, answers_item))
    # check that only one of both categroy is possible
    return(list(set(answers)))

def process_all_traffic_answers(traffic_df, purchased_cat, filters_def_dict, type_filters):
    """
    wrapper function to process all answers from the dataset.
    """
    # assuming you have eliminated all invalid urls
    idx = traffic_df.index.values
    urlsList = traffic_df["RequestUrl"].values
    for i, url in zip(idx, urlsList):
        ans_d = {}
        try:
            propgroup_list, filters, dict_dict_answers = filters_answers_per_requestURL(url)
            for f in filters: 
                tmp_new_ans = process_answers_filter(f, dict_dict_answers, filters_def_dict, type_filters)
                ans_d.update({f: tmp_new_ans})
            if bool(ans_d):
                traffic_df.loc[i, "answer"] = [ans_d]
            else:
                traffic_df.loc[i, "answer"] = [dict()]
        except TypeError:
            # print('error in filters_answers_per_requestURL')
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