from typing import List, Dict
from collections import defaultdict
import os
import glob

# find min and max value in a string like '0.1/0.5 0.5/1.0 1.0/2.0'
def find_min_max_in_string(s):
    elements = s.split()

    numbers = []
    for element in elements:
        numbers.extend(map(float, element.split('/')))

    min_value = min(numbers)
    max_value = max(numbers)

    return f"{min_value}/{max_value}"

# generate sac2spec terminal cmd
def gen_sac2spec_cmd(seis_array:Dict,command_dict: Dict, xc_param: Dict) -> List[str]:
    """
    Generate sac 2 spec commands for both 1 and 3 components
    """
    win_len = xc_param['win_len']
    whiten_types = {'OFF': 0, 'BEFORE': 1, 'AFTER': 2, 'BOTH': 3}
    cuda_whiten = xc_param['whiten']
    cuda_whiten = whiten_types.get(cuda_whiten, 2)
    normalize_types = {'OFF': 0, 'RUN-ABS': 1, 'ONE-BIT': 2}
    cuda_normalize = xc_param['normalize']
    cuda_normalize = normalize_types.get(cuda_normalize, 1)
    band_info = xc_param['bands']
    whiten_band = find_min_max_in_string(band_info)
    skip_step = xc_param['skip_step']

    # iterate over the gpu_dir in the work_dir
    output_dir = xc_param['output_dir']
    sac_spec_list_dir = os.path.join(output_dir, 'sac_spec_list')
    
    # the filter file is in the output_dir
    filter_file = os.path.join(output_dir, 'butterworth_filter.txt')
    
    comopnent_num = len(seis_array['component_list'])

    param_set = [
    '-L', str(win_len),
    '-W', str(cuda_whiten),
    '-N', str(cuda_normalize),
    '-F', whiten_band,
    '-Q', str(skip_step),
    '-B', filter_file]
    
    cmd_list = []
    # iterate over the gpu_dir in the work_dir, each gpu_dir contains sac and spec lists
    sac_lists = glob.glob(os.path.join(sac_spec_list_dir,'sac_list_*.txt'))
    gpu_num = len(sac_lists)
    for sac_list in sac_lists:
        info = os.path.basename(sac_list).split('.')[0]
        gpu_id = info.split('_')[-1]
        spec_list = os.path.join(sac_spec_list_dir, f"spec_list_{gpu_id}.txt")
        local_param_set = [command_dict['sac2spec'],'-I', sac_list,'-O', spec_list] 
        local_param_set += ['-C',str(comopnent_num)]
        local_param_set += param_set + ['-G', str(gpu_id),'-T',str(gpu_num)]
        cmd = ' '.join(local_param_set)
        cmd_list.append(cmd)

    cmd_list_dir = os.path.join(output_dir, 'cmd_list')
    os.makedirs(cmd_list_dir, exist_ok=True)
    cmd_file = os.path.join(cmd_list_dir, 'sac2spec_cmds.txt')
    with open(cmd_file, 'w') as f:
        f.write('\n'.join(cmd_list))
    return cmd_list

# generate cross correlation terminal cmd
def gen_xc_cmd(command_dict: Dict, xc_param: Dict, seis_array2: Dict) -> List[str]:
    output_dir = xc_param['output_dir']
    xc_list_dir = os.path.join(output_dir, 'xc_list')

    # check if there is double array situation
    single_array_situation = seis_array2['sac_dir'] == 'NONE'

    # this param_set is for both xc_mono and xc_dual
    param_set = [
        '-O', os.path.join(output_dir, 'ncf'), '-C', str(xc_param['max_lag'])]

    # set an empty cmd_list
    cmd_list = []

    # iterate over directories with time_label in the xc_list_dir
    for time_label in os.listdir(xc_list_dir):
        time_dir = os.path.join(xc_list_dir, time_label)

        src_files = sorted(
            glob.glob(os.path.join(time_dir, 'src_speclist_*.txt')))
        sta_files = sorted(
            glob.glob(os.path.join(time_dir, 'sta_speclist_*.txt')))

        if sta_files and not single_array_situation:
            for src in src_files:
                for sta in sta_files:
                    input_set = ['-A', src, '-B', sta]
                    cmd = command_dict['xc_dual'] + ' ' + \
                        ' '.join(input_set) + ' ' + ' '.join(param_set)
                    cmd_list.append(cmd)

        if single_array_situation:
            for i in range(len(src_files)):
                for j in range(i, len(src_files)):
                    src1 = src_files[i]
                    src2 = src_files[j]
                    # using xc_dual if two sub array are different
                    if src1 != src2:
                        input_set = ['-A', src1, '-B', src2]
                        cmd = command_dict['xc_dual'] + ' ' + \
                            ' '.join(input_set) + ' ' + ' '.join(param_set)
                        cmd_list.append(cmd)
                    else:
                        input_set = ['-A', src1]
                        cmd = command_dict['xc_mono'] + ' ' + \
                            ' '.join(input_set) + ' ' + ' '.join(param_set)
                        cmd_list.append(cmd)

    cmd_list_dir = os.path.join(output_dir, 'cmd_list')
    os.makedirs(cmd_list_dir, exist_ok=True)
    with open(os.path.join(cmd_list_dir, 'xc_cmds.txt'), 'w') as f:
        f.write('\n'.join(cmd_list))

    return cmd_list

# generate terminal stack cmd

def gen_stack_cmd(command_dict: Dict, xc_param: Dict) -> List[str]:
    """
    Generate commands for stack
    """
    cmd_list = []
    output_dir = xc_param['output_dir']
    normalize = xc_param['stack_norm']
    command = command_dict['stack']

    stack_list_dir = os.path.join(output_dir, 'stack_list')
    stack_dir = os.path.join(output_dir, 'stack')
    cmd_list_dir = os.path.join(output_dir, 'cmd_list')

    stack_lists = [os.path.join(stack_list_dir, fname)
                   for fname in os.listdir(stack_list_dir)]

    os.makedirs(stack_dir, exist_ok=True)

    for stack_list in stack_lists:
        stack_list_name = os.path.basename(stack_list)
        # split the file name to get important label, info[0] is the station pair name and info[1] is for component pair
        info = stack_list_name.split('.')
        output_fname = f"{info[0]}.{info[1]}.ncf.sac"
        output_path = os.path.join(stack_dir, info[0], output_fname)

        if normalize:
            cmd = f"{command} -I {stack_list} -O {output_path} "
        else:
            cmd = f"{command} -I {stack_list} -O {output_path} -A"

        cmd_list.append(cmd)

    os.makedirs(cmd_list_dir, exist_ok=True)
    cmd_list_path = os.path.join(cmd_list_dir, 'stack_cmds.txt')
    with open(cmd_list_path, 'w') as f:
        f.write('\n'.join(cmd_list))
    return cmd_list

# generate terminal rotate cmd


def gen_rotate_cmd(command_dict: Dict, xc_param: Dict) -> List[str]:
    cmd_list = []
    output_dir = xc_param['output_dir']
    rotate_list_dir = os.path.join(output_dir, 'rotate_list')
    cmd_list_dir = os.path.join(output_dir, 'cmd_list')
    command = command_dict['rotate']

    for rotate_list in os.listdir(rotate_list_dir):
        # Get first key in the enz_group dictionary
        target_dir = os.path.join(rotate_list_dir, rotate_list)
        inlist = os.path.join(target_dir, 'enz_list.txt')
        outlist = os.path.join(target_dir, 'rtz_list.txt')
        cmd = f"{command} -I {inlist} -O {outlist}"
        cmd_list.append(cmd)

    os.makedirs(cmd_list_dir, exist_ok=True)
    with open(os.path.join(cmd_list_dir, 'rotate_cmds.txt'), 'w') as f:
        f.write('\n'.join(cmd_list))

    return cmd_list
