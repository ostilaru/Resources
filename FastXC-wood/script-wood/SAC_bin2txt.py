from obspy import read
from obspy.io.sac import SACTrace, SACStream

def convert_binary_to_text_sac(binary_file, text_file):
    # 读取二进制型的 SAC 文件
    st = read(binary_file, format='SAC')

    # 将每个 Trace 的数据转换为文本格式
    sac_stream = SACStream()
    for trace in st:
        sac_trace = SACTrace.from_obspy_trace(trace)
        sac_stream.append(sac_trace)

    # 写入为文本型的 SAC 文件
    sac_stream.write(text_file)

if __name__ == "__main__":
    binary_sac_file = '/home/wood/github/data/output/sac_bin/T1.KCD01.20071002063000.00.BHZ.sac'
    text_sac_file = '/home/wood/github/data/output/sac_txt/T1.KCD01.20071002063000.00.BHZ.sac'

    convert_binary_to_text_sac(binary_sac_file, text_sac_file)
