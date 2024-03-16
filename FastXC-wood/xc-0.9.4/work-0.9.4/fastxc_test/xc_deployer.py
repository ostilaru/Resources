import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import  Dict
import os
import re

def setup_logger(log_file_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Or another level

    # Use a thread-safe RotatingFileHandler
    handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class GPUWorker:
    def __init__(self, gpu_id, function, max_workers):
        self.gpu_id = gpu_id
        self.function = function
        self.tasks = queue.Queue()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.should_shutdown = False  # Control stop submit mission

    def _process_tasks(self):    
        while not self.should_shutdown:
            try:
                # Wait for a new task and get it
                task = self.tasks.get(timeout=2)  # timeout of 1 second

                # Submit the task to the executor
                self.executor.submit(self.function, task)

                # Mark the task as done in the queue
                self.tasks.task_done()
            except queue.Empty:
                # If the queue is empty, continue the loop to check the shutdown flag again
                continue
            except KeyboardInterrupt:
                self.should_shutdown = True
                break             

        self.executor.shutdown(wait=True)

    def submit(self, task):
        # if should_shutdonw no more task add ib
        if not self.should_shutdown:
            self.executor.submit(self.function, task)
        else:
            pass

    def shutdown(self):
        # set should_shutdown Ture, avoid submitting more tasks
        self.should_shutdown =True
        
        # adding None to all task line, tell them to shut down
        self.executor.shutdown(wait=True)
            
        print(f"GPU Worker {self.gpu_id} has been shut down.")

class MultiGPUProcessor:
    def __init__(self, function, gpu_ids, max_workers_per_gpu):
        self.workers = {gpu_id: GPUWorker(gpu_id, function, max_workers) for gpu_id, max_workers in
                        zip(gpu_ids, max_workers_per_gpu)}

    def distribute_tasks(self, gpu_cmd_dict):
        for gpu_id, cmd_list in gpu_cmd_dict.items():
            for cmd in cmd_list:
                self.workers[gpu_id].submit(cmd)

    def shutdown(self):
        for worker in self.workers.values():
            worker.shutdown()


def xc_cmd_deployer(xc_param: Dict, mem_info: Dict, executing: Dict):
    output_dir = xc_param['output_dir']
    max_workers_per_gpu = mem_info['gpu_task_num']
    gpu_mem = mem_info['gpu_mem_info']
    gpu_list = mem_info['gpu_list']
    debug = executing['debug']
    xc_cmd_file = os.path.join(output_dir, 'cmd_list', 'xc_cmds.txt')  
    logger = setup_logger(executing['log_file_path'])
    
    with open(xc_cmd_file, 'r') as f:
        xc_cmd_list = [line.strip() for line in f if line.strip()]

    
    # Define the function that will be executed by the workers
    def run_cmd(exe_cmd: str):
        cmd_info = exe_cmd.split()[2]
        time_label = cmd_info.split('/')[-2]
        gpu_id = re.findall(r"-G (\d+)",exe_cmd)
        gpu_id = gpu_id[0]
        
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{formatted_time}] GPU no.{gpu_id} Doing Cross Correlation on {time_label}')
        
        result = subprocess.run(exe_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # write the output to the log file
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)

    def print_cmd(exe_cmd: str):
        print(exe_cmd)
        cmd_info = exe_cmd.split()[2]
        time_label = cmd_info.split('/')[-2]
        gpu_id = re.findall(r"-G (\d+)",exe_cmd)
        gpu_id = gpu_id[0]
        
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{formatted_time}] GPU no.{gpu_id} Doing Cross Correlation on {time_label}')
        
        logger.debug(f"Command: {exe_cmd}")
    
    run_or_print = print_cmd if debug else run_cmd

    # Ensure max_workers_per_gpu and gpu_mem_info are of same length
    num_gpus = len(gpu_mem)
    gpu_mem_info = dict(zip(gpu_list, gpu_mem))
    if len(max_workers_per_gpu) < num_gpus:
        # If max_workers_per_gpu is shorter, extend it with 1s
        max_workers_per_gpu.extend([1] * (num_gpus - len(max_workers_per_gpu)))
    elif len(max_workers_per_gpu) > num_gpus:
        # If max_workers_per_gpu is longer, only keep the smallest ones
        max_workers_per_gpu = sorted(max_workers_per_gpu)[:num_gpus]

    # sort the gpu_mem_info by memory size
    sorted_gpus = sorted(gpu_mem_info.keys())
    max_workers_per_gpu = [x for _, x in
                           sorted(zip(gpu_mem_info.values(), max_workers_per_gpu), key=lambda pair: pair[0])]
    
    # Launch the workers
    processor = MultiGPUProcessor(run_or_print, sorted_gpus, max_workers_per_gpu)

    total_mem = sum(gpu_mem_info.values())
    gpu_portions = {gpu_id: mem / total_mem for gpu_id, mem in gpu_mem_info.items()}

    cmd_counts = {gpu_id: round(len(xc_cmd_list) * portion) for gpu_id, portion in gpu_portions.items()}

    cmd_index = 0
    xc_flag = executing['do_xc']

    for gpu_id, count in cmd_counts.items():
        for _ in range(count):
            if cmd_index < len(xc_cmd_list):
                # adding -G gpu_id to the command
                cmd = f"{xc_cmd_list[cmd_index]} -G {gpu_id}"
                if xc_flag:
                    cmd = f"{cmd} -X"
                processor.workers[gpu_id].submit(cmd)
                cmd_index += 1
            else:
                break
    
    # if there are still commands left, add them to the first gpu
    while cmd_index < len(xc_cmd_list):
        for gpu_id in sorted_gpus:
            cmd = f"{xc_cmd_list[cmd_index]} -G {gpu_id}"
            print(xc_flag==True)
            if xc_flag:
                cmd = f"{cmd} -X"
            processor.workers[gpu_id].submit(cmd)
            cmd_index += 1
            if cmd_index >= len(xc_cmd_list):
                break

    # after all the commands are submitted, shutdown the workers
    processor.shutdown()
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Finish doing Cross Correlation !!!.\n")
