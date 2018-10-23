#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
from parser import parse_query_string

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield list(map(int,iterable[ndx:min(ndx + n, l)]))


def keep_only_useful_URLs(df):
    new = df.copy()
    for i in df.index.values:
        if i%1000 == 0:
            print(i)
            print(len(new))
        url = new.loc[i, "RequestUrl"]
        if not bool(parse_query_string(url)):
            new = new.drop(i) # eliminate the row if the parser returns empty dict
    return(new)