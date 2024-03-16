from SeisHandler import SeisArray
from typing import Dict, List, Tuple
from pandas import Timestamp
import numpy as np
import os

def gen_seis_file_group_list(seis_array_info_dict_list: List[Dict]) -> List:
    """
    generating seis_file_group_list based on info_list (parsed from congig.ini file)
    """
    seis_file_group_list = []

    for seis_array_info_dict in seis_array_info_dict_list:
        if seis_array_info_dict['sac_dir'].upper() == 'NONE':
            continue

        seis_array_dir = seis_array_info_dict['sac_dir']
        pattern = seis_array_info_dict['pattern']
        sta_list_path = seis_array_info_dict['sta_list']
        time_list = [seis_array_info_dict['start'], seis_array_info_dict['end']]
        component_list = seis_array_info_dict['component_list']
        sta_list = []

        if sta_list_path != 'NONE':
            with open(sta_list_path, 'r') as f:
                for line in f.readlines():
                    sta_list.append(line.strip())

        # core function genrating seis_file_group_list
        seis_array = SeisArray(seis_array_dir, pattern)
        seis_array.match()
        if sta_list:
            criteria = {'station': sta_list,'time': time_list, 'component': component_list}
        else:
            print("[Warning]: Sta_list is empty will not use as a criteria")
            criteria = {'time': time_list, 'component': component_list}
        
        print('\n')
        
        seis_array.filter(criteria)
        seis_array.group(labels=['station', 'time'], sort_labels=['component'], filtered=True)

        if seis_array.files_group:
            seis_file_group_list.append(seis_array.files_group)

    return seis_file_group_list

def gen_sac2spec_list(seis_file_group_list: List[Dict[Tuple[str, Timestamp], Dict[str, List[str]]]],
                      spec_dir: str,component_flag:int) -> List[Dict[str, List[str]]]:
    sorted_seis_file_group_list = sorted(
        seis_file_group_list, key=lambda x: list(x.keys())[0][0])
    sac_spec_pairs = []
    for i, seis_file_group in enumerate(sorted_seis_file_group_list):
        array_id = str(i + 1)

        for sta_time in seis_file_group.keys():
            components = seis_file_group[sta_time]['component']
            if len(components)!=component_flag:
                continue
            paths = seis_file_group[sta_time]['path']
            station = list(sta_time)[0]
            time = list(sta_time)[1]
            time = time.strftime("%Y.%j.%H%M")

            sac_paths = []
            spec_paths = []

            for component, sac_path in zip(components, paths):
                spec_name = station + '.' + time + '.' + component + '.segspec'
                spec_path = os.path.join(
                    spec_dir, 'array' + str(array_id), time, spec_name)

                sac_paths.append(sac_path)
                spec_paths.append(spec_path)

            sac_spec_pair = {'sac': sac_paths, 'spec': spec_paths}
            sac_spec_pairs.append(sac_spec_pair)

    return sac_spec_pairs


def distribute_tasks(gpu_mem_info: Dict[int, int], num_tasks: int) -> Dict[int, int]:
    """
    Function to distribute tasks among available GPUs based on their memory size.

    Parameters:
    gpu_mem_info (Dict[int, int]): A dictionary with GPU ID as the key and corresponding GPU memory as the value.
    num_tasks (int): The total number of tasks to be distributed.

    Returns:
    Dict[int, int]: A dictionary with GPU ID as the key and the number of tasks assigned to it as the value.
    """

    total_memory = sum(gpu_mem_info.values())
    gpu_tasks = {}
    for gpu, memory in gpu_mem_info.items():
        gpu_tasks[gpu] = int(np.floor(num_tasks * memory / total_memory))

    # Calculate the number of tasks already assigned
    assigned_tasks = sum(gpu_tasks.values())

    # If there are remaining tasks, assign them to the GPU with the largest memory
    remaining_tasks = num_tasks - assigned_tasks
    print('remaining_tasks',remaining_tasks)
    if remaining_tasks > 0:
        largest_memory_gpu = max(gpu_mem_info, key=gpu_mem_info.get)
        gpu_tasks[largest_memory_gpu] += remaining_tasks

    # Assertion to check that all tasks have been assigned
    assert sum(gpu_tasks.values()) == num_tasks, "Not all tasks were correctly assigned!"
    return gpu_tasks


def write_sac2spec_list(sac_spec_pairs: List[Dict[str, List[str]]], gpu_mem_info: Dict[int, int],
                        sac_spec_list_dir: str,component_flag:int):
    sac_spec_pair_num = len(sac_spec_pairs)
    gpu_tasks = distribute_tasks(gpu_mem_info, sac_spec_pair_num)

    # Iterate over the sac_spec_pairs, allocate them to different gpus and create files
    pair_index = 0
    for gpu, task_num in gpu_tasks.items():
        sac_list_file = os.path.join(sac_spec_list_dir, 'sac_list_' + str(gpu) + '.txt')
        spec_list_file = os.path.join(sac_spec_list_dir, 'spec_list_' + str(gpu) + '.txt')

        f_sac = open(sac_list_file, 'w')
        f_spec = open(spec_list_file, 'w')

        for _ in range(task_num):
            if pair_index < sac_spec_pair_num:
                components_num = len(sac_spec_pairs[pair_index]['sac'])
                if components_num != component_flag:
                    continue
                for component_index in range(components_num):
                    f_sac.write(sac_spec_pairs[pair_index]['sac'][component_index]+'\n')
                    f_spec.write(sac_spec_pairs[pair_index]['spec'][component_index]+'\n')
                
                pair_index += 1
            else:
                break
            
        f_sac.close()
        f_spec.close()


# Main function in this scripts
def gen_sac2spec_list_dir(seis_array1: Dict, seis_array2: Dict, xc_param: Dict, mem_info: Dict):
    # step 0: extract parameters from input
    output_dir = xc_param['output_dir']
    sac_spec_list_dir = os.path.join(output_dir, 'sac_spec_list')
    os.makedirs(sac_spec_list_dir, exist_ok=True)
    spec_dir = os.path.join(output_dir, 'segspec')
    components_flag = len(seis_array1['component_list'])
    gpu_mem = mem_info['gpu_mem_info']
    gpu_list = mem_info['gpu_list']
    
    # step 1 generateing the info_dict of gpus
    gpu_mem_info = dict(zip(gpu_list, gpu_mem))
    # step 2 generating sac file list group
    seis_file_group = gen_seis_file_group_list([seis_array1, seis_array2])
    # step 3 creating output spec path for each sac file
    sac_spec_pairs = gen_sac2spec_list(seis_file_group, spec_dir,components_flag)
    # step 4 write into .txt file
    write_sac2spec_list(sac_spec_pairs, gpu_mem_info, sac_spec_list_dir,components_flag)
