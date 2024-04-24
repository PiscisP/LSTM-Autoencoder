import numpy as np
import scipy.signal as signal

'''
def segment_signal(data, fs, unit_factor=1, seg_seconds=1):
    # 算法返回与数据长度相同的标签，1：关机；2：空转；3：工作，0：非常态。

    thresh_peak = 0.05 * unit_factor
    thresh_working = 0.05 * unit_factor
    thresh_len_working = 10
    thresh_idle = 0.01 * unit_factor
    thresh_no_load_min = 0.03 * unit_factor
    thresh_no_load_max = 0.1 * unit_factor

    segment_len = 1
    # 确保长度是 fs * segment_len 的整数倍
    length = len(data) // (fs * segment_len) * (fs * segment_len)
    current_signal_segment = data[:length]

    # 重塑信号
    signal_reshaped = current_signal_segment.reshape(-1, fs * segment_len)
    signal_max = np.max(signal_reshaped, axis=1)
    signal_peak = signal.medfilt(signal_max, 9)

    # 使用find_peaks进行峰值检测
    peaks, _ = signal.find_peaks(signal_peak, height=thresh_peak)

    # 确定工作区间和非常态区间
    working_slices = []
    unstable_slices = []
    for peak in peaks:
        begin = max(0, peak - np.argmax(signal_peak[peak::-1] < thresh_working))
        end = peak + np.argmax(signal_peak[peak:] < thresh_working)
        if (end - begin) > thresh_len_working:
            working_slices.append((begin, end))
        else:
            unstable_slices.append((begin, end))

    # 标签赋值
    labels = np.zeros_like(signal_max)
    for begin, end in working_slices:
        labels[begin:end] = 3
    labels[(labels == 0) & (signal_max < thresh_idle)] = 1
    labels[(labels == 0) & ((signal_max > thresh_no_load_min) & (signal_max < thresh_no_load_max))] = 2

    # 合并间隔过小的非常态区间
    for begin, end in unstable_slices:
        labels[begin:end] = 0
    unstable_slices = get_slices(labels, 0)
    if len(unstable_slices) >= 2:
        tmp_list = []
        a = unstable_slices.pop(0)
        while unstable_slices:
            b = unstable_slices.pop(0)
            if (b[0] - a[1]) < thresh_len_working:
                a = [a[0], b[1]]
            else:
                tmp_list.append(a)
                a = b
        tmp_list.append(a)
        unstable_slices = tmp_list
    for begin, end in unstable_slices:
        labels[begin:end] = 0

    # 标签扩展
    labels = np.repeat(labels, fs * segment_len)

    slices = [
        get_slices(labels, 0),
        get_slices(labels, 1),
        get_slices(labels, 2),
        get_slices(labels, 3),
    ]

    return labels, slices

'''
def segment_signal(data, unit_factor=1, previous_working=False):
    """
    对信号进行分段处理，并根据各段的最大值平均与阈值比较来设置标签。
    通过previous_working跟踪前一段数据的工作状态来保持工作状态的连续性。
    超过最大阈值0.2的段被标记为异常。

    参数:
    data : np.array
        输入的信号数据。
    unit_factor : float, 可选
        用于调整阈值的系数，默认为1。
    previous_working : bool, 可选
        前一数据段是否处于工作状态。

    返回:
    labels : np.array
        标签数组，标识各数据点的状态。
    working : bool
        当前数据段是否有工作状态。
    """
    
    segment_len = len(data) // 5
    segments_max = [np.max(data[i * segment_len:(i + 1) * segment_len]) for i in range(5)]
    max_average = np.mean(segments_max)  # 计算最大值的平均
    
    # labels 1：关机；2：空转；3：工作，0：非常态。
    # 定义不同状态的阈值
    max_threshold = 0.2  # 设定的异常值阈值
    thresh_peak = 0.06 * unit_factor #工作状态阈值
    thresh_continuation = 0.045 * unit_factor #继续工作状态阈值
    thresh_idle = 0.01 * unit_factor
    thresh_no_load_min = 0.03 * unit_factor
    thresh_no_load_max = 0.07 * unit_factor
    
    
    labels = np.zeros(len(data))
    working = previous_working

    # 检查是否有异常值超过max_threshold
    if any(max_val > max_threshold for max_val in segments_max):
        labels.fill(0)  # 标记整个段为异常状态
        working = False  # 可选：标记为非工作状态
    elif working and max_average >= thresh_continuation:
        labels.fill(3)
    elif not working and max_average >= thresh_peak:
        labels.fill(3)
        working = True
    elif max_average < thresh_idle:
        labels.fill(1)
        working = False
    elif thresh_no_load_min <= max_average <= thresh_no_load_max:
        labels.fill(2)
        working = False
    else:
        # 默认处理为异常状态，这可能需要进一步细化
        labels.fill(0)
        working = False

    return labels, working

def segment_signal_conveyor(data, fs, unit_factor=1):
    # 算法返回与数据长度相同的标签，1：关机；2：空转；3：工作，0：非常态。

    thresh_peak = 0.06 * unit_factor
    thresh_working = 0.055 * unit_factor
    thresh_len_working = 10
    thresh_idle = 0.01 * unit_factor
    thresh_no_load_min = 0.03 * unit_factor
    thresh_no_load_max = 0.07 * unit_factor

    segment_len = 1
    # 确保长度是 fs * segment_len 的整数倍
    length = len(data) // (fs * segment_len) * (fs * segment_len)
    current_signal_segment = data[:length]

    # 重塑信号
    signal_reshaped = current_signal_segment.reshape(-1, fs * segment_len)
    signal_max = np.max(signal_reshaped, axis=1)
    signal_peak = signal.medfilt(signal_max, 9)

    # 使用find_peaks进行峰值检测
    peaks, _ = signal.find_peaks(signal_peak, height=thresh_peak)

    # 确定工作区间和非常态区间
    working_slices = []
    unstable_slices = []
    for peak in peaks:
        begin = max(0, peak - np.argmax(signal_peak[peak::-1] < thresh_working))
        end = peak + np.argmax(signal_peak[peak:] < thresh_working)
        if (end - begin) > thresh_len_working:
            working_slices.append((begin, end))
        else:
            unstable_slices.append((begin, end))

    # 标签赋值
    labels = np.zeros_like(signal_max)
    for begin, end in working_slices:
        labels[begin:end] = 3
    labels[(labels == 0) & (signal_max < thresh_idle)] = 1
    labels[(labels == 0) & ((signal_max > thresh_no_load_min) & (signal_max < thresh_no_load_max))] = 2

    # 合并间隔过小的非常态区间
    for begin, end in unstable_slices:
        labels[begin:end] = 0
    unstable_slices = get_slices(labels, 0)
    if len(unstable_slices) >= 2:
        tmp_list = []
        a = unstable_slices.pop(0)
        while unstable_slices:
            b = unstable_slices.pop(0)
            if (b[0] - a[1]) < thresh_len_working:
                a = [a[0], b[1]]
            else:
                tmp_list.append(a)
                a = b
        tmp_list.append(a)
        unstable_slices = tmp_list
    for begin, end in unstable_slices:
        labels[begin:end] = 0

    # 标签扩展
    labels = np.repeat(labels, fs * segment_len)

    slices = [
        get_slices(labels, 0),
        get_slices(labels, 1),
        get_slices(labels, 2),
        get_slices(labels, 3),
    ]

    return labels, slices
    

def get_slices(labels, mode):
    # 获取指定状态的数据起止索引
    # mode：1：关机；2：空转；3：工作，0：非常态。
    tmp = np.where(labels == mode, 1, 0)
    tmp = np.diff(tmp, prepend=0, append=0)
    starts = np.where(tmp == 1)[0]
    ends = np.where(tmp == -1)[0]
    slices = [(starts[i], ends[i]) for i in range(len(starts))]
    return slices

def slices_trans(slices):
    return sorted([(s[0], s[1], num) for num, sli in enumerate(slices) for s in sli], key=lambda x: x[0])


def main():
    from os.path import join
    data_dir = 'dextro_data'
    fs = 2000
    sn = '00001-20231024-0014'
    date = '2024-01-04'

    ref_data = np.fromfile(join(data_dir, sn, f'{sn}_10_{date}.bin'), dtype='float32')
    labels, slices = segment_signal(ref_data, fs)

    slices = slices_trans(slices)
    print(slices)

if __name__ == '__main__':
    main()
