#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data Science Lab Project - FALL 2018
Mélanie Bernhardt - Mélanie Gaillochet - Laura Manduchi

This parser was written by Digitec to 
helps us to parse the RequestUrl from 
the traffic data table.
"""

# PropertyGroup is for example Width, Table Properties or RAM
# PropertyDefinition is for example Shape, Size or Ram Type
# PropertyDefinitions have a Type. 
# - Either its just a value then we find it in the PropertyValue
# - If we have a fixed set of options they are PropertyDefinitionOption objects.
# PropertyDefinitionOption is actually a choice of Shape or RAM Type
# A Property links a ProductType a ProductDefinition and a Product Group
# A Property itself can also take one or multiple values

# The parser handles the query string on pages that support filtering.
# The output is represents the currently selected options 

import json

def handle_opt(opt, result):
    """
    This section handles properties that either directly have a primitive type value (bool, multidimensional properties)
    or Properties that link a PropertyGroup and PropertyDefinition which again map directly to a primitive type value.
    The Properties that directly have a primitive value will have the value in the coloumn PropertyValue
    Also the properties with a list of possible values will have the value in the coloumn PropertyValue
    """
    parts = opt.split('|')
    for part in parts:
        # Boolean properties
        if part[0] == 't' or part[0] == 'f':
            property_id = int(part[1:])
            
            if 'Property' not in result:
                result['Property'] = dict()
            result['Property'][property_id] = 1 if part[0] == 't' else 0
        
        # Multidimensional properties
        if part[0] == 'm':
            prefix, values = part.split(':')
            property_id = int(prefix[1:])
            values = values.split(',')
            
            if 'Property' not in result:
                result['Property'] = dict()
            result['Property'][property_id] = values
        
        # Single Value properties
        if part[0] == 'v':
            prefix, value = part.split(':')
            if '~' in prefix:
                continue
            property_group_id, property_definition_id = prefix[1:].split('-')
            
            if 'PropertyGroup' not in result:
                result['PropertyGroup'] = dict()
            if property_group_id not in result:
                result['PropertyGroup'][property_group_id] = dict()
            if 'PropertyDefinition' not in result['PropertyGroup'][property_group_id]:
                result['PropertyGroup'][property_group_id]['PropertyDefinition'] = dict()
                
            result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id] = value
            
    return result

def handle_bra(section, result):
    '''
    This section handles Brands.
    Example:
    https://www.galaxus.ch/de/s1/producttype/notebook-6?bra=1|47&tagIds=614
    ProductType 6 (Notebook)
    Selected are the Brands 1 (ASUS) and 47 (Apple)
    '''
    brand_ids = section.split('|')
    result['Brands'] = brand_ids
    return result

def handle_rng_rou(section, result):
    '''
    This section handles ranges.
    I.e. Table Width for example.
    The min and max values are to be found in the coloumn PropertyValue
    '''
    tuples = section.split('|')
    for tup in tuples:
        prefix, suffix = tup.split(':')
        if '~' in prefix:
            continue
        property_group_id, property_definition_id = prefix.split('-')
        minimum, maximum = suffix.split(',')
        if 'PropertyGroup' not in result:
            result['PropertyGroup'] = dict()
        if property_group_id not in result:
            result['PropertyGroup'][property_group_id] = dict()
        if 'PropertyDefinition' not in result['PropertyGroup'][property_group_id]:
            result['PropertyGroup'][property_group_id]['PropertyDefinition'] = dict()
        result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id] = dict()
        result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id]['Min'] = minimum
        result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id]['Max'] = maximum
    return result

def handle_pdo(section, result):
    '''
    This section handles ProductPropertyOptions.
    These represent the fixed sets of options there are on a specific product type
    Example:
    https://www.galaxus.ch/de/s1/producttype/notebook-6?pdo=13-6885:277226&tagIds=614
    ProductType 6 (Notebook)
    Selected is the PropertyDefinitionOption 277226 (Windows 10 Pro)
    This is a option of the PropertyDefinition 6885 (Windows Version)
    This again is a definition in the PropertyGroup 13 (Operating System)
    '''
    parts = section.split('|')
    for part in parts:
        prefix, property_definition_option_id = part.split(':')
        if '~' in prefix:
            continue
        property_group_id, property_definition_id = prefix.split('-')
        if 'PropertyGroup' not in result:
            result['PropertyGroup'] = dict()
        if property_group_id not in result:
            result['PropertyGroup'][property_group_id] = dict()
        if 'PropertyDefinition' not in result['PropertyGroup'][property_group_id]:
            result['PropertyGroup'][property_group_id]['PropertyDefinition'] = dict()
        if property_definition_id not in result['PropertyGroup'][property_group_id]['PropertyDefinition']:
            result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id] = dict()
        if 'PropertyDefinitionOptionIds' in result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id]:
            result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id]['PropertyDefinitionOptionIds'].append(property_definition_option_id)
        else:
            result['PropertyGroup'][property_group_id]['PropertyDefinition'][property_definition_id]['PropertyDefinitionOptionIds'] = [property_definition_option_id]

    return result
    
def handle_section(section, result):
    if (section[:3] == 'opt'):
        result = handle_opt(section[4:], result)
    if (section[:3] == 'bra'):
        result = handle_bra(section[4:], result)
    if (section[:3] == 'rng' or section[:3] == 'rou'):
        result = handle_rng_rou(section[4:], result)
    if (section[:3] == 'pdo'):
        result = handle_pdo(section[4:], result)
    return result

def parse_query_string(query_string):
    try:
        sections = query_string.split('&')
        result = dict()
        for section in sections:
            result = handle_section(section, result)
        #print(json.dumps(result, indent=2))
        return(result)
    except:
       # print({})
        return ('{}')

example_qstring = 'opt=t44|m141:1,-2,3.14159,4,5|v3125-598080:6&bra=3301&nov=1:-30|2:15&rng=12-123:0.667,5|12-124:-2.7,1.414&rou=11-125:4,6.283|11-126:2.7,4&pdo=3126-598081:132|344-576:298&p=7667:1928|5123:1815&rfb=1&sale=1&pr=1&sr=1'
parse_query_string(example_qstring)