from obspy import read
import numpy as np
import math
import os
import multiprocessing
from tqdm import tqdm
from datetime import datetime

def sac2dat(sac,target_dir):
    # Read in sac file
    st = read(sac)
    tr = st[0] # only one trace in sac file
    
    # Extract header information
    delta = tr.stats.delta
    npts = tr.stats.sac.npts
    stla, stlo, stel = tr.stats.sac.stla, tr.stats.sac.stlo, 0
    evla, evlo, evel = tr.stats.sac.evla, tr.stats.sac.evlo, 0
    
    # Data Processing
    data = tr.data
    half_length = math.ceil(npts / 2)
    data_neg = data[:half_length][::-1]  # Negative half
    data_pos = data[(half_length - 1):]   # Positive half
    times = np.arange(0, half_length*delta, delta)
    
    basename, _ = os.path.splitext(os.path.basename(sac))
    new_filename = basename + '.dat'
    outpath = os.path.join(target_dir, new_filename)   
    # Write output file
    
    with open(outpath, 'w') as f:
        # Write station and event location information
        f.write(f"{stlo:.7e} {stla:.7e} {stel:.7e}\n")
        f.write(f"{evlo:.7e} {evla:.7e} {evel:.7e}\n")
        
        # Write time, negative half, and positive half data
        for i, time in enumerate(times):
            neg = data_neg[i] if i < len(data_neg) else 0  # Pad with zeros if necessary
            pos = data_pos[i] if i < len(data_pos) else 0  # Pad with zeros if necessary
            f.write(f"{time:.7e} {neg:.7e} {pos:.7e}\n")
    


def sac2dat_main(sac_path, input_root, output_root):
    # Calculate output directory based on the relative path of the SAC file
    relative_path = os.path.relpath(os.path.dirname(sac_path), input_root)
    # outdir = os.path.join(output_root, relative_path)
    outdir = os.path.join(output_root)
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Process the SAC file
    sac2dat(sac_path, outdir)

def update_progress(*a):
    pbar.update()

if __name__ == '__main__':
    source_dir = '/mnt/f/ncf_test/rtz'
    desitnation_dir = '/mnt/f/ncf_test/dat'
    tasks = [] 

    print(f"[{datetime.now()}]:Finding all .sac files")
    for root,_,files in os.walk(source_dir):
        for file in files:
            if file.endswith('.sac'):
                info = file.split('.')[1]
                if info!='T-T':
                    continue
                sac_path = os.path.join(root,file)
                task = (sac_path,source_dir,desitnation_dir)
                tasks.append(task)
    print(f"[{datetime.now()}]:Find out all .sac files")
                
    # Initialize tqdm progress bar
    pbar = tqdm(total=len(tasks))
    
    # Initialize multiprocessing Pool
    pool = multiprocessing.Pool(processes=30)
    
    # Apply tasks to the pool
    for task in tasks:
        pool.apply_async(sac2dat_main, (task[0],task[1],task[2]),callback=update_progress)
        #sac2dat_main(task[0],task[1],task[2])

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Close the tqdm progress bar
    pbar.close()

    # write a list file
    dat_list = os.path.join(desitnation_dir,'dat_list.txt')
    with open(dat_list,'w') as f:
        for _,_,files in os.walk(desitnation_dir):
            for file in files:
                if file.endswith('.dat'):
                    f.write(file+'\n')
    
