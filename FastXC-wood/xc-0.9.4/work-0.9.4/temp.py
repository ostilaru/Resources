#!/bin/python
import os
import glob

data_dir = '/mnt/f/tl-wf'

all_sta = []
for sta_dir in glob.glob(data_dir+'/*'):
    sta  = os.path.basename(sta_dir)
    all_sta.append(sta)

filtered_sta = []
for sta in all_sta:
    if sta[0:2]=='01':
        filtered_sta.append(sta)

for sta in filtered_sta:
    print(sta)

with open('./sta_list.txt','w+') as f:
    for sta in filtered_sta:
        f.write(sta+'\n')
