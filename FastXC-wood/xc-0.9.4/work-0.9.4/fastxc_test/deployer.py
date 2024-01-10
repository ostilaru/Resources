import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List, Dict
import logging
from threading import Event
from multiprocessing import Pool
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import time
from datetime import datetime
import os

# PART0: UTILS
def setup_logger(log_file_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Or another level

    # Use a thread-safe RotatingFileHandler
    handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_or_print_cmd(cmd: str, debug: bool = False, logger=None):
    if debug:
        # Simply print the command if debug is True.
        print(f"Command: {cmd}")
    elif logger:
        # If a logger is provided, capture output and log it.
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                
                logger.info(result.stdout)  # record the output
            else:
                logger.error(result.stderr)  # record the error as command failed

        except Exception as e:        
            logger.error(f"An unexpected error occurred: {e}")
    else:
        # If no logger and not in debug mode, run the command without capturing output.
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def read_cmds_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as f:
            cmds = [line.strip() for line in f if line.strip()]
        return cmds
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        return []
    
# PART1: SAC2SPEC
stop_thread = False # 1. definea the shared flag

def check_and_print_file_count(base_dir:str, stop_event: threading.Event, interval:int=10):
    """
    Periodically check and print the number of files in the given directory (recursively).

    Args:
    - base_dir (str): Directory to check.
    - interval (int): Time interval (in seconds) to wait between checks.
    - stop_event (threading.Event): An event that signals the thread to stop.
    """
    while not stop_event.is_set():
        file_count = sum([len(files) for _, _, files in os.walk(base_dir)])
        print(f"[{datetime.now()}]: Current number of spectrums written out: {file_count}")
        time.sleep(interval)

def sac2spec_cmd_deployer(xc_param: Dict, executing: Dict):
    log_file_path = executing['log_file_path']
    output_dir = xc_param['output_dir']
    debug = executing['debug']
    parallel = executing['parallel']
    segspec_dir = os.path.join(output_dir, "segspec")
    cmd_list_file = os.path.join(output_dir, 'cmd_list', 'sac2spec_cmds.txt')
    logger = setup_logger(log_file_path)
    cmds = read_cmds_from_file(cmd_list_file)
    
    # Create an Event object
    stop_event = Event()

    # Start the thread to check and print file count
    check_thread = threading.Thread(target=check_and_print_file_count, args=(segspec_dir, stop_event))
    check_thread.start()
    
    if parallel:
        # parallel execution
        with ThreadPoolExecutor(max_workers=len(cmds)) as executor:
            for cmd in cmds:
                executor.submit(run_or_print_cmd, cmd, debug, logger)
    else:
        # serial execution
        for cmd in cmds:
            run_or_print_cmd(cmd, debug, logger)

    # Set the event to signal the thread to stop
    stop_event.set()
    check_thread.join()
    
    # Print the final file count
    file_count = sum([len(files) for _, _, files in os.walk(segspec_dir)])
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Total number of spectrums written out: {file_count}")
    
    print(f"\n[{formatted_time}]: Finish doing SAC2SPEC !.\n")

# PART2: STACKING AND ROTATING
def worker(cmd_debug_pair):
    cmd, debug = cmd_debug_pair
    run_or_print_cmd(cmd, debug)
    return True  # Just to indicate that the task was processed. Can be omitted if not needed.

def execute_in_batches(pool, cmds: List[str], batch_size: int, debug, information):
    start_index = 0
    total_cmds = len(cmds)
    
    while start_index < total_cmds:
        end_index = start_index + batch_size
        current_batch = [(cmd, debug) for cmd in cmds[start_index:end_index]]

        # You can use the pool.map or pool.imap to process the batch
        for _ in tqdm(pool.imap(worker, current_batch), total=len(current_batch), desc=information):
            pass

        start_index = end_index

def stack_cmd_deployer(xc_param: Dict, executing: Dict):
    output_dir = xc_param['output_dir']
    log_file_path = executing['log_file_path']
    cmd_list_file = os.path.join(output_dir, 'cmd_list', 'stack_cmds.txt')
    debug = executing['debug']
    parallel = executing['parallel']
    threads = executing['threads'] # Default to CPU count if not provided.
    stack_cmds = read_cmds_from_file(cmd_list_file)
    logger = setup_logger(log_file_path)
    
    if parallel:
        # parallel execution
        with Pool(processes=threads) as pool:
            execute_in_batches(pool, stack_cmds, 5000, debug, "[Stacking]")
    else:
        # serial execution
        for cmd in tqdm(stack_cmds, desc="[Stacking]"):
            run_or_print_cmd(cmd, debug, logger)
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Finish doing STACK .\n")

def rotate_cmd_deployer(xc_param: Dict, executing: Dict):
    output_dir = xc_param['output_dir']
    log_file_path = executing['log_file_path']
    cmd_list_file = os.path.join(output_dir, 'cmd_list', 'rotate_cmds.txt')
    debug = executing['debug']
    parallel = executing['parallel']
    threads = executing['threads']
    rotate_cmds = read_cmds_from_file(cmd_list_file)
    logger = setup_logger(log_file_path)
    
    if parallel:
        # 并行执行
        with Pool(processes=threads) as pool:
            execute_in_batches(pool, rotate_cmds, 5000, debug, "[Rotating]")
    else:
        # 串行执行
        for cmd in tqdm(rotate_cmds, desc="[Rotating]"):
            run_or_print_cmd(cmd, debug, logger)
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Finish doing ROTATE !!!.\n")
