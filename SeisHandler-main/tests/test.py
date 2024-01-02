import sys
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from SeisHandler import SeisArray as sa

# matching files
path_to_array_dir = current_dir / 'example' / 'testArray'

# define the pattern of file
seis_file_pattern = '{home}/{*}/{*}/{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.{suffix}'

# create an example of SeisArray
my_array = sa(array_dir=path_to_array_dir, pattern=seis_file_pattern)

# matching file using multi threads
my_array.match(threads=20)

# display the first 3 files matched
file_info = my_array.files
print("[Member of my_array.files]")
for i in range(0,3):
    file = file_info[i]
    print(file,'\n')
print("[************************]\n")

# define a file filtering criteria
component_list = ['X']
time_list = ['2018-01-01 00:00:00', '2018-01-07 00:00:00']
criteria = {'component': component_list, 'time': time_list}

# apply the file filtering criteria
my_array.filter(criteria=criteria, threads=20)
file_info = my_array.filtered_files
# display the first 3 file after filtering
print("[Member of my_array.files After filtering]")
for i in range(0,3):
    file = file_info[i]
    print(file,'\n')
print("[************************]\n")

# group the file by labels
my_array.group(labels=['station', 'time'], filtered=True)

# display the first group after grouping files
print("First group of files after grouping files by label")
for i in range(0,1):
    station_time = list(my_array.files_group.keys())[0]
    file_group = my_array.files_group[station_time]
    print(file_group)
print("**********************************\n")

# generate a multi-level dictionary for virtual array
print("Reorganizing files by label")
my_array.organize(label_order=['station','time'],filtered=True,output_type='path')
for station in my_array.virtual_array:
    for time in my_array.virtual_array[station]:
        print(station,time,my_array.virtual_array[station][time])
        break
    break