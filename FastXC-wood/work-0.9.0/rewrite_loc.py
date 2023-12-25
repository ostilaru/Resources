import os
import glob
import obspy
from multiprocessing import Pool
from tqdm import tqdm

def check_loc(sac_file, coords):
    fname = os.path.basename(sac_file)
    kstnm = fname.split('_')[0]
    try:
        true_stla = coords[kstnm][0]
        true_stlo = coords[kstnm][1]
    except KeyError:
        print(f"Coordinates for station {kstnm} not found.")
        return

    sac = obspy.read(sac_file)  # read only the header

    # try:
    #     old_stla = sac[0].stats.sac.stla
    #     old_stlo = sac[0].stats.sac.stlo
    # except AttributeError:
    #     print(f"'stla' or 'stlo' not found in SAC header for file {sac_file}.")
    #     return
    
    # print(f"old_stla:{old_stla} || new_stla:{true_stla}")
    # print(f"old_stlo:{old_stlo} || new_stlo:{true_stlo}")
    
    # Uncomment the following lines if you want to write the new coordinates to the SAC file
    sac[0].stats.sac.stla = true_stla
    sac[0].stats.sac.stlo = true_stlo
    sac.write(sac_file, format='SAC')
    
    return



def get_all_files(target_dir):
    all_paths = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_paths.append(file_path)
    
    filtered_path = []
    for path in all_paths:
        path_fname = os.path.basename(path)
        kstnm = path_fname.split('_')[0]
        if kstnm[0]=='F':
            filtered_path.append(path)
    
    return filtered_path


def parse_file_to_dict(file_path):
    # Function to parse a line of data into a dictionary
    def parse_line_to_dict(line):
        parts = line.strip().split()
        if len(parts) == 5:  # Assuming each line has 5 pieces of data
            return parts[0], (float(parts[2]), float(parts[3]))
        else:
            return None, None  # In case a line doesn't have exactly 5 parts

    # Parsing each line in the file to a dictionary with ID as key and (Longitude, Latitude) as value
    data_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = parse_line_to_dict(line)
                if key:  # Only add if the line was parsed successfully
                    data_dict[key] = value
    except IOError:
        print(f"Could not read file: {file_path}")
    return data_dict

if __name__ == '__main__':
    # You would call the function like this:
    file_path = '/storage/HOME/yaolab/data/wjx/datas/liying/2018-2020-Noise_HFP.txt'
    loc_infos = parse_file_to_dict(file_path)
    
    data_path = '/storage/HOME/yaolab/data/wjx/datas/liying/2018-2020'
    all_files = get_all_files(data_path)
    
    # for file in all_files:
    #     print(file)
        
    from functools import partial
    check_loc_with_coords = partial(check_loc, coords=loc_infos)

    with Pool(10) as p:
        r = list(tqdm(p.imap(check_loc_with_coords, all_files), total=len(all_files)))
