""" Data Science Lab Project - FALL 2018
Mélanie Bernhardt - Mélanie Gaillochet - Laura Manduchi

This file just contains the helper function to 
merge several data files from several teacher runs.
"""

import tensorlayer as tl
import numpy as np

def process_all_teacher_files(list_filenames, outputname='_tmp.npy'):
    """This function is used to merge several data files from 
    several teacher runs.
    
    Args:
        list_filenames: list of filenames to be merged
        outputname: filename where to save the merged data
    
    Example:
        list_filenames = ['s200_p2a0.0_p3a0.0_pidk0.0_a1.0_tmp.npy', 's200_p2a0.2_p3a0.1_pidk0.1_a1.0_tmp.npy']
        process_all_teacher_files(list_filenames, outputname='test.npy')  
    """
    tmp = tl.files.load_npy_to_any(name=list_filenames[0])
    act = tmp['act']
    state = tmp['state_list']
    for d in list_filenames[1:]:
        tmp = tl.files.load_npy_to_any(name=d)
        act = np.append(act, tmp['act'])
        state = np.append(state, tmp['state_list'])
    tl.files.save_any_to_npy(save_dict={'state_list': state, 'act': act}, name=outputname)