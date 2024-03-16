from obspy import read, Stream
import os

def read_sac_list(sac_list_file):
    st = Stream()

    with open(sac_list_file, 'r') as f:
        for line in f:
            file_path = line.strip()

            if os.path.exists(file_path):
                trace = read(file_path, format='SAC')[0]
                st += trace
                print(f"Read SAC file: {file_path}")
            else:
                print(f"File not found: {file_path}")

    return st

def write_trace_info_to_txt(stream, output_file):
    with open(output_file, 'w') as output:
        for trace in stream:
            output.write(str(trace) + '\n')

if __name__ == "__main__":
    sac_list_file = 'FastXC-wood/data/sac_list_file_woodwood/20071002063000.txt'
    output_txt_file = 'FastXC-wood/data/output/file.txt'

    stream_data = read_sac_list(sac_list_file)
    write_trace_info_to_txt(stream_data, output_txt_file)
