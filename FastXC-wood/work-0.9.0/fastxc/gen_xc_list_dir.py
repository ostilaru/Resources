# Purpose: Generating the workdir for spec cross-correlation
from SeisHandler import SeisArray
import logging
from contextlib import contextmanager
from typing import List, Dict
from datetime import datetime
import numpy as np
import os
import sys

'''
[PART 0] Functions for calculating the number of station that can be processed
'''

def find_optimal_transform_length(input_length):
    optimal_length = input_length
    while True:
        remaining_factor = optimal_length
        prime_factors = [2, 3, 5, 7]
        for i in range(4):
            while (remaining_factor > 1) and (remaining_factor % prime_factors[i] == 0):
                remaining_factor = remaining_factor / prime_factors[i]
        if remaining_factor == 1:
            break
        optimal_length = optimal_length + 1
    return optimal_length

def find_largest_transform_length_below(input_length):
    if input_length <= 1:
        print("Error: Invalid Argument", file=sys.stderr)
        return 0
    optimal_length = input_length - 1
    while optimal_length > 0:
        remaining_factor = optimal_length
        prime_factors = [2, 3, 5, 7]
        for i in range(4):
            while (remaining_factor > 1) and (remaining_factor % prime_factors[i] == 0):
                remaining_factor = remaining_factor / prime_factors[i]
        if remaining_factor == 1:
            return optimal_length
        optimal_length = optimal_length - 1
    print("Error: No Suitable Number Found", file=sys.stderr)
    return 0

def cal_seg_size(seg_len, npts, delta):
    seg_npts = seg_len / delta
    nfft = find_optimal_transform_length(int(seg_npts))

    if nfft > npts:
        nfft = npts
        nfft = find_largest_transform_length_below(nfft)

    num_seg = nfft
    nfft_2x = num_seg * 2
    num_spec = nfft_2x // 2 + 1
    num_step = npts // seg_npts

    return 30 + num_step * num_spec * np.dtype('float32').itemsize * 2 # GPU reading in size

def estimate_max_station_num(mem_info,seis_array_1):
    npts = mem_info['npts']
    delta = mem_info['delta']
    seg_len = mem_info['seg_len']
    gpu_mem_info = mem_info['gpu_mem_info']
    cpu_mem_info = mem_info['cpu_mem_info']
    redundant_ratio = mem_info['redundant_ratio']
    gpu_list = mem_info['gpu_list']
    gpu_task_num = mem_info['gpu_task_num']
    
    component_list = seis_array_1['component_list']
    component_num  = len(component_list) 

    # Calculate size of a segment in bytes
    size_of_segment = cal_seg_size(seg_len, npts, delta)

    # Convert memory info from GigaBytes to Bytes
    gpu_mem_info_bytes = [mem * (1024 ** 3) for mem in gpu_mem_info]

    # Calculate available memory based on abundant ratio
    gpu_available_mem_bytes = [
        mem * redundant_ratio / task_num for mem, task_num in zip(gpu_mem_info_bytes, gpu_task_num)]
    cpu_available_mem_bytes = cpu_mem_info * (1024 ** 3) * redundant_ratio

    # Calculate the number of spectra that each GPU can handle
    num_of_spectra_per_gpu = [int(mem // size_of_segment)
                              for mem in gpu_available_mem_bytes]

    # Calculate total number of spectra that CPU can handle
    num_of_spectra_host = int(cpu_available_mem_bytes // len(gpu_list) // size_of_segment)

    # Final number of spectra is minimum of CPU and GPU capacity
    num_of_spectra = min(min(num_of_spectra_per_gpu), num_of_spectra_host)

    # 2 below is a redundant message to make sure that we have enough space on GPU
    num_of_sta = int(num_of_spectra / (3 * component_num))
    
    return num_of_sta

'''
[PART 1]: Functions for split the seismic array
'''
@contextmanager
def suppress_logging():
    prev_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(prev_level)


# step0:  Split array using max_sta_num
def split_array(segspec_dir:str,max_sta_num)->List[List[str]]:
    pattern = '{home}/{network}/{*}/{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.segspec'
    
    with suppress_logging():
        spec_array = SeisArray(array_dir=segspec_dir, pattern=pattern)
        spec_array.match()
        spec_array.group(labels=['network','station'], filtered=False)
        array_files = spec_array.files_group
    
    src_stations = []
    sta_stations = []
    for file_info in array_files:
        network,station = file_info
        if network == 'array1':
            src_stations.append(station)
        if network == 'array2':
            sta_stations.append(station)
    
    src_groups = [src_stations[i:i+max_sta_num] for i in range(0, len(src_stations), max_sta_num)]
    sta_groups = [sta_stations[i:i+max_sta_num] for i in range(0, len(sta_stations), max_sta_num)]
    
    return src_groups, sta_groups


'''
[PART 2] Functions for generating the workdir
'''
def find_sublist_index(lst, target):
    for index, sublist in enumerate(lst):
        if target in sublist:
            return index
    return 999  # if not found, return 999

def write_xc_list(xc_list_dir: str, segspec_dir:str, src_groups:List[List[str]], sta_groups: List[List[str]], time_list: List[List[str]]):
    
    pattern = '{home}/{network}/{*}/{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.segspec'
    
    # dealing with spectrum array
    with suppress_logging():
        spec_array = SeisArray(array_dir=segspec_dir, pattern=pattern)
        spec_array.match()
    
    criteria = {'time':time_list}
    spec_array.filter(criteria)
    spec_array.group(labels=['time'], filtered=True)
    files_group_dict = spec_array.files_group
    
    all_write_files = {}
    for time_stamp,files_info in files_group_dict.items():
        time_label = time_stamp.strftime('%Y.%j.%H%M')
        network_info = files_info['network']
        sta_info = files_info['station']
        filepaths = files_info['path']
        
        for network, station, file_path in zip(network_info,sta_info,filepaths):
            group_idx = 0
            if network=='array1':
                group_idx = find_sublist_index(src_groups, station)    
                net_label = 'src'
            if network=='array2':
                group_idx = find_sublist_index(sta_groups, station)
                net_label = 'sta'
            
            list_file_name = net_label+'_speclist_'+str(group_idx)+'.txt'
            list_file_path = os.path.join(xc_list_dir,time_label,list_file_name)
            
            if list_file_path not in all_write_files:
                all_write_files[list_file_path] = []
                
            all_write_files[list_file_path].append(file_path)
    
    # sort file paths in each file_list
    for file_list_path, paths in all_write_files.items():
        
        # sort file list by station name
        def sort_key_func(filepath):
            return os.path.basename(filepath).split('.')[0]
        
        paths.sort(key=sort_key_func)

        dir_name = os.path.dirname(file_list_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_list_path, 'w') as f:
            for path in paths:
                f.write(path + '\n')

'''
[PART 3] main function in this file
'''
def gen_xc_list_dir(xc_param: Dict, mem_info: Dict, seis_array1:Dict):
    # Get parameters
    output_dir = xc_param['output_dir']
    
    segspec_dir = os.path.join(output_dir, 'segspec')

    xc_list_dir = os.path.join(output_dir, 'xc_list')

    max_sta_num = estimate_max_station_num(mem_info,seis_array1)
    
    src_groups, sta_groups = split_array(segspec_dir,max_sta_num)
    
    time_list = [seis_array1['start'], seis_array1['end']]
    
    write_xc_list(xc_list_dir, segspec_dir, src_groups, sta_groups, time_list)

