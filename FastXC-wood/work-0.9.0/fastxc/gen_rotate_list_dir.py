from typing import Dict, List
import os
from itertools import product

def gen_rotate_list_dir(seis_array1: Dict, seis_array2: Dict, xc_param: Dict) -> bool:

    flag = seis_array2['sac_dir']
    component_list_1 = seis_array1['component_list']
    component_list_2 = component_list_1 if flag == 'NONE' else seis_array2['component_list']

    # if len!=3 ,exit
    if len(component_list_1) != 3 or len(component_list_2) != 3:
        print('Only 3 components situation is supported in Rotate!')
        return False

    output_dir = xc_param['output_dir']
    stack_dir = os.path.join(output_dir, 'stack')
    rtz_dir = os.path.join(output_dir, 'rtz')
    rotate_list_dir = os.path.join(output_dir, 'rotate_list')

    mapping = {component_list_1[i]: val for i,
               val in enumerate(['E', 'N', 'Z'])}
    mapping.update(
        {component_list_2[i]: val for i, val in enumerate(['E', 'N', 'Z'])})

    component_pair_order = ['EE', 'EN', 'EZ',
                            'NE', 'NN', 'NZ', 'ZE', 'ZN', 'ZZ']

    for sta_pair in os.listdir(stack_dir):
        sta_pair_path = os.path.join(stack_dir, sta_pair)
        enz_sac_num = len(os.listdir(sta_pair_path))
        if enz_sac_num != 9:
            continue
        enz_group = {}
        for component_pair in product(component_list_1, component_list_2):
            component1, component2 = component_pair
            file_name = f'{sta_pair}.{component1}-{component2}.ncf.sac'
            file_path = os.path.join(sta_pair_path, file_name)

            component_pair = f'{mapping[component1]}{mapping[component2]}'
            enz_group.update({component_pair: file_path})

        # sort the files by order of component_pair_order
        enz_group = {key: enz_group[key] for key in component_pair_order}

        rotate_dir = os.path.join(rotate_list_dir, sta_pair)
        os.makedirs(rotate_dir, exist_ok=True)

        in_list = os.path.join(rotate_dir, 'enz_list.txt')
        out_list = os.path.join(rotate_dir, 'rtz_list.txt')

        # Write the input stack file paths
        with open(in_list, 'w') as f:
            f.write('\n'.join(enz_group.values()))

        # Write the output RTZ file paths
        with open(out_list, 'w') as f:
            for component_pair in ['R-R', 'R-T', 'R-Z', 'T-R', 'T-T', 'T-Z', 'Z-R', 'Z-T', 'Z-Z']:
                outpath = os.path.join(
                    rtz_dir, sta_pair, f'{sta_pair}.{component_pair}.ncf.sac')
                f.write(outpath + '\n')
    
    return True
