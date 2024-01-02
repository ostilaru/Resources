from obspy import read

# 读取 SAC 文件
st = read('/home/wood/github/data/20071002063000/T1.KCD01.20071002063000.00.BHZ.sac', format='SAC')
print(st)

#st.plot()

print(len(st))

print(st[0].stats)

# # 获取第一个 Trace 的波形数据
# first_trace_data = st[0].data

# # 打印波形数据的前几个样本点
# print("First few samples of waveform data:", first_trace_data[:10])
