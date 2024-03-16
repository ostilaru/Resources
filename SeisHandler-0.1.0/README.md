# Description of SeisHandler

SeisHandler is a Python package for organizing seismic data.
It does not read any data inside files but matches and organizes
files by their names. It is a tool for organizing seismic data using
several specific labels.

# Design philosophy

It has been a problem that has confused people for a long time about organizing seismic array data. A seismic array is a group of seismometers deployed in a certain area for a time and recording ground motion signals. A seismic array sometimes comprises instruments of the same kind and has the same characteristics.

The style of organizing seismic data is something like philosophy. The organizing style depicts how the user will use the data. For example, if the data is organized in a '/data_home/station/time/data' style,
researchers may take advantage of this style to do some research using a single-station method, receiver function e.g.
If organizing data in a '/data_home/time/station/data' style,
The researcher will want to show the result at different times, like velocity monitoring.

Thus, it is difficult for all methods to be applied to the same data set before re-organizing the path of data. Moreover, this obstacle is the usage of some very obstacle methods. I designed this class to
organize the data in a virtual style inspired by the thought of virtualization.
(The motor can achieve the effect of a gearbox and reduce the use of mechanical components through a virtual control
system)
In this package, developers can reshape the path of data by changing the order of labels. Furthermore, if they design their own programs based on this and all the seismic developers follow this style, the program will be more flexible and powerful. It will be easier
for everyone's method to be applied to different data sets very simply.

## Installation

pre-requirement pandas:

```
pip install pandas
```

Install this package

```
pip install SeisHandler.tar.gz
```

## Usage

```
from SeisHandler import SeisArray as sa

array_dir = 'path/to/array/dir'
pattern = '{file}/{path}/{pattern}{with}{label}'

# matching files
my_array = sa(array_dir=array_dir, pattern=pattern)
my_array.match(num_threads=4)

# filter files by criteria, output is my_array.filtered_files
criteria = { 'label': [label_list],''}
# e.g 
''' 
    criteria = {'component': ['X'], 'time': [
    '2018-01-01 00:00:00', '2018-01-07 00:00:00']}
'''
my_array.filter(criteria=criteria,threads=4)


# group seis files by labels
# the output is a dictionary, whose key is a comined label
my_array.group(['station','time'])

# re-orgainze files by labels
# the output is a multi-level dictionary, whose key is a label
my_array.organize(label_order=['station','time'],filtered=True,output_type='path')
for station in my_array.virtual_array:
    for time in my_array.virtual_array[station]:
        print(my_array.virtual_array[station][time])
        print('\n')
```

## Example

```
from SeisHandler import SeisArray as sa



# matching files
path_to_array_dir = 'abspath/to/demo/data'
seis_file_pattern = '{home}/{*}/{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.{suffix}'
my_array = sa(array_dir=path_to_array_dir, pattern=seis_file_pattern)
my_array.match(threads=20)

'''
for file in my_array.files:
    print(file)
'''

criteria = {'component': ['X'], 'time': [
    '2018-01-01 00:00:00', '2018-01-07 00:00:00']}
my_array.filter(criteria=criteria, threads=20)

my_array.group(labels=['station', 'time'], filtered=True)

'''
for station_time in my_array.files_group:
    print(my_array.files_group[station_time])
    print('\n')
'''

# generate a multi-level dictionary for virtual array
my_array.organize(label_order=['station','time'],filtered=True,output_type='path')
for station in my_array.virtual_array:
    for time in my_array.virtual_array[station]:
        print(my_array.virtual_array[station][time])
        print('\n')
```

## Changelog

### v0.1.0 (2023-07-10)

- The First version.

## Contact

If you have any questions, issues, or suggestions for improvement,
feel free to open an issue in the repository,
or directly reach out to me at wkh16@mail.ustc.edu.cn.

## License

SeisHandler is licensed under the MIT License. See [LICENSE](LICENSE) file for details.


