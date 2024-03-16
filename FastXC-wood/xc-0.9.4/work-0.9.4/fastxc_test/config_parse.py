import os
import configparser
from datetime import datetime


def parse_and_check_ini_file(file_path):
    def convert_type(key, value):
        # Convert boolean values
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False

        # Convert integer values
        if key in ['npts', 'seg_len', 'cpu_mem_info', 'threads']:
            return int(value)

        # Convert float values
        if key in ['redundant_ratio','delta']:
            return float(value)

        # Convert lists of integers
        if key in ['gpu_list', 'gpu_task_num', 'gpu_mem_info']:
            return [int(x) for x in value.split(',')]

        if key in ['component_list']:
            return value.split(',')

        # Other keys remain as strings
        return value

    # Parsing the INI file
    config = configparser.ConfigParser()
    config.read(file_path)

    # Check each seis_array
    for i in range(1, 3):
        section_name = f"SeisArray_{i}"
        print(f"[Info]: checking {section_name}...")

        # check the existence of sac dir
        sac_dir = config.get(section_name, 'sac_dir')
        if sac_dir != "NONE" and not os.path.exists(sac_dir):
            print(f"[Error]:  Path {sac_dir} is not exists.")
        elif sac_dir == "NONE":
            continue

        # check the time
        start_time = config.get(section_name, 'start')
        end_time = config.get(section_name, 'end')
        start_time_obj = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time_obj = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        if start_time_obj >= end_time_obj:
            raise ValueError(  # noqa
                f"[Error] start time {start_time} should be earlier than end time {end_time}.")

        # check sta_list
        sta_list = config.get(section_name, 'sta_list')
        if sta_list != "NONE" and not os.path.exists(sta_list):
            raise FileNotFoundError(f"[Error]  Path {sta_list} is not exists.")

    # check whiten and normalization
    whiten_value = config.get('Parameters', 'whiten')
    if whiten_value not in ['OFF', 'BEFORE', 'AFTER', 'BOTH']:
        raise ValueError(  # noqa
            f"[Error] Invalid whiten value {whiten_value} in section Parameters.")

    normalize_value = config.get('Parameters', 'normalize')
    if normalize_value not in ['OFF', 'RUN-ABS', 'ONE-BIT']:
        raise ValueError(  # noqa
            f"[Error] Invalid normalize value {normalize_value} in section Parameters.")

    # check the command executable file
    print("checking command line tools ...")
    command_section = config['Command']
    for command in command_section:
        path = command_section[command]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[Error]  Path {path} is not exists.")

    # checking if componet_list_1 and component_list_2 are the same length
    component_list_1 = config.get('SeisArray_1', 'component_list')
    component_list_2 = config.get('SeisArray_2', 'component_list')
    if config.get('SeisArray_2', 'sac_dir') != 'NONE' and len(component_list_1.split(',')) != len(component_list_2.split(',')):
        raise ValueError(  # noqa
            f"[Error] component_list_1 and component_list_2 should be the same length.")

    print("[INFO]: Finish checking.")

    SeisArray_1 = {key: convert_type(key, value)
                   for key, value in config['SeisArray_1'].items()}
    SeisArray_2 = {key: convert_type(key, value)
                   for key, value in config['SeisArray_2'].items()}
    Command = dict(config['Command'])  # Assuming all command paths are strings
    Parameters = {key: convert_type(key, value)
                  for key, value in config['Parameters'].items()}
    mem_info = {key: convert_type(key, value)
                for key, value in config['mem_info'].items()}
    executing = {key: convert_type(key, value)
                 for key, value in config['executing'].items()}
     
     # change daul_array to true if sacdir in seis_array_2 is not NONE
    if SeisArray_2['sac_dir'] != 'NONE':
        Parameters['dual_array'] = True
    else:
        Parameters['dual_array'] = False

    return SeisArray_1, SeisArray_2, Parameters, Command, mem_info, executing
