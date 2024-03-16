from obspy import read

# 读取 SAC 文件
st = read('/home/woodwood/hpc/station_2/AAKH/2018/AAKH.2018.001.0000.U.sac', format='SAC')
print(st)

#st.plot()

print(len(st))

print(st[0].stats)

# # 获取第一个 Trace 的波形数据
# first_trace_data = st[0].data

# # 打印波形数据的前几个样本点
# print("First few samples of waveform data:", first_trace_data[:10])
