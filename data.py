import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv
import os

def segment_signal(data, fs, unit_factor=1):
    # labels 1：关机；2：空转；3：工作，0：非常态。
    thresh_peak = 0.05 * unit_factor
    thresh_working = 0.05 * unit_factor
    thresh_idle = 0.01 * unit_factor
    thresh_no_load_min = 0.03 * unit_factor
    thresh_no_load_max = 0.1 * unit_factor

    length = len(data) // (fs * 4) * (fs * 4)
    current_signal_segment = data[:length]
    signal_reshaped = current_signal_segment.reshape(-1, fs * 4)
    signal_max = np.max(signal_reshaped, axis=1)
    signal_peak = signal.medfilt(signal_max, 9)
    peaks, _ = signal.find_peaks(signal_peak, height=thresh_peak)
    
    working_slices = []
    labels = np.zeros_like(signal_max)
    for peak in peaks:
        begin = max(0, peak - np.argmax(signal_peak[peak::-1] < thresh_working))
        end = peak + np.argmax(signal_peak[peak:] < thresh_working)
        working_slices.append((begin * (fs * 4), end * (fs * 4)))
        labels[begin:end] = 3

    labels[(labels == 0) & (signal_max < thresh_idle)] = 1
    labels[(labels == 0) & ((signal_max > thresh_no_load_min) & (signal_max < thresh_no_load_max))] = 2
    labels = np.repeat(labels, fs * 4)

    return labels, working_slices

def preprocess_data(raw_data, fs ,file_path):
    labels, working_slices = segment_signal(raw_data, fs)
    if labels.size == 0:
        print("No labels to process.")
        return np.array([])

    working_data = []
    for start, end in working_slices:
        working_data.append(raw_data[start:end])

    if not working_data:
        return np.array([])

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start Index', 'End Index'])
        for start, end in working_slices:
            writer.writerow([start, end])

    return np.concatenate(working_data)

#读取csv
def read_slices(csv_file_path):
    slices = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            start, end = map(int, row)
            slices.append((start, end))
    return slices

def extract_segments(vibration_data, slices, fs):
    segments = []
    for start, end in slices:
        data_start = start 
        data_end = end 
        if data_end > len(vibration_data):
            print(f"Slice {start}-{end} is out of bounds.")
            continue
        segment = vibration_data[data_start:data_end]

        #消除振幅超过0.7的片段
        if np.max(np.abs(segment)) <= 0.7:
            segments.append(segment)
    
    return segments

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def main_process(fs, current_file_path, vibration_file_path, csv_file_path, unit_factor=1, step_length=10000):
    current_data = np.fromfile(current_file_path, dtype=np.float32)
    vibration_data= np.fromfile(vibration_file_path, dtype=np.float32)
    '''
    #打印电流图像
    plt.figure(figsize=(15, 6))
    plt.plot(current_data)
    plt.title('Current Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
    
    print(f"Current data shape: {current_data.shape}")
    #打印原始震动图像
    plt.figure(figsize=(15, 6))
    plt.plot(vibration_data)
    plt.title('Vibration Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
    print(f"Vibration data shape: {vibration_data.shape}")
    '''

    #提取出工作状态的索引并存入csv文件
    working_data = preprocess_data(current_data, fs, csv_file_path)
    print(f"Working data shape: {working_data.shape}")

    #读取csv文件中的索引
    slices = read_slices(csv_file_path)

    #根据索引提取震动数据
    segments = extract_segments(vibration_data, slices, fs)
    print(f"Number of extracted segments: {len(segments)}")
    all_data = np.concatenate(segments)
    
    #打印工作状态的震动图像
    plt.figure(figsize=(15, 6))
    plt.plot(all_data)
    plt.title('Vibration Segments Combined')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
    
    #进行标准化
    normalized_data = normalize_data(all_data)

    #重塑为三维并存储以用来训练 形状为（样本数，时间步长 = 10000，特征数 = 1）
    num_steps = len(normalized_data) // step_length
    reshaped_data = normalized_data[:num_steps * step_length].reshape(num_steps, step_length, 1)
    print(f"Shape of reshaped data: {reshaped_data.shape}")
    npz_file_path = os.path.join(base_path, 'data', f'vibration_{date}.npz')
    np.savez(npz_file_path, reshaped_data)


# 设置参数
fs = 2000  # 采样频率

#只需要改日期就可以切换文件
date = '2023-12-31'
# 获取脚本文件当前位置的路径
base_path = os.path.dirname(__file__)
# 用os.path.join来构建相对路径
vibration_file_path = os.path.join(base_path, 'data', 'Vibration', f'00001-20231024-0014_01_{date}.bin')
current_file_path = os.path.join(base_path, 'data', 'Current', f'00001-20231024-0014_10_{date}.bin')
csv_file_path = os.path.join(base_path, 'data', 'csv',f'working_data_{date}.csv')


def main():
    # 执行主处理流程
    main_process(fs, current_file_path, vibration_file_path, csv_file_path)

if __name__ == "__main__":
    main()

