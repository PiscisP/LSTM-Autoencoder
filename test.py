import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os

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


#模型结构
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_dim, output_dim, batch_first=True)
        self.decoder1 = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        x, _ = self.encoder1(x)
        x, _ = self.encoder2(x)
        x, _ = self.decoder1(x)
        x = self.output(x)
        return x

# 创建模型实例
def load_model(model_path, input_dim, hidden_dim, output_dim, device):
    model = LSTMAutoencoder(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 推理计算损失
def predict_loss(data_loader, model, device):
    """
    使用加载的模型对数据进行推理，并计算预测损失。

    参数:
    data_loader : DataLoader
        包含待测试数据的数据加载器。
    model : LSTMAutoencoder
        加载好的LSTM自编码器模型。
    device : torch.device
        指定计算设备。

    返回:
    average_loss : float
        计算得到的平均损失值。
    """
    criterion = nn.MSELoss()
    total_loss = 0
    losses = []  # 用于存储每个批次的损失

    with torch.no_grad():
        for data in data_loader:
            data_tensor = data[0].to(device)
            outputs = model(data_tensor)
            loss = criterion(outputs, data_tensor)
            total_loss += loss.item()
            losses.append(loss.item())

    average_loss = total_loss / len(data_loader)
    '''
    # 绘制损失随时间的曲线
    plt.plot(losses)
    plt.title('Loss over time')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.savefig(save_path)  # 保存图像
    plt.show()
    plt.close()
    '''
    return average_loss



def test(model, data_segment, device, z_score_threshold=6):
    """
    对给定的数据段进行预处理和异常检测测试。使用Z-score方法来识别异常值，并通过模型预测数据的损失。

    参数:
    model : LSTMAutoencoder
        使用的模型。
    data_segment : np.array
        要测试的数据段。
    device : torch.device
        设备，如CPU或GPU，用于模型计算。
    z_score_threshold : float
        Z-score阈值，用于识别和过滤异常值。

    返回:
    loss : float 或 None
        如果数据段没有异常，返回模型的损失值；如果有异常，则返回None。
    """
    
    # 计算 Z-score
    mean = np.mean(data_segment)
    std = np.std(data_segment)
    if std == 0:
        return None  # 如果标准差为0，则跳过该段
    
    #使用Z-score过滤
    z_scores = np.abs((data_segment - mean) / std)
    if np.any(z_scores > z_score_threshold):
        print("Segment contains outliers based on Z-score, filtering out")
        return None
    
    # 转换为 PyTorch tensor
    data_tensor = torch.tensor(data_segment, dtype=torch.float32)

    # 标准化数据段
    normalized_tensor = (data_tensor - mean) / std if std > 0 else data_tensor

    # 添加额外的批次和特征维度
    normalized_tensor = normalized_tensor.unsqueeze(0).unsqueeze(2)

    # 移动到正确的设备
    normalized_tensor = normalized_tensor.to(device)

    # 创建 DataLoader
    data_loader = DataLoader(TensorDataset(normalized_tensor, normalized_tensor), batch_size=1, shuffle=False)

    # 调用 predict_loss 或其他处理函数继续处理
    return predict_loss(data_loader, model, device)

def process_segment(current_data, vibration_data, i, segment_length, model, device, working, accumulated_vibration, losses):
    """
    处理单个数据段，根据电流和振动信号的数据，对模型进行测试，并更新损失和工作状态。

    参数:
    current_data : np.array
        电流数据的数组。
    vibration_data : np.array
        振动数据的数组。
    i : int
        当前处理的起始索引。
    segment_length : int
        处理的数据段长度。
    model : torch.nn.Module
        用于测试的模型。
    device : torch.device
        指定模型运行在哪个设备上（如CPU或GPU）。
    working : bool
        表示上一个段是否处于工作状态。
    accumulated_vibration : np.array
        累积的振动数据。
    losses : list
        记录每个工作段的损失值。

    返回:
    labels: np.array 返回新的标签数组
    working: bool 是否处于工作状态
    accumulated_vibration: np.array 返回累积的振动数据
    """
    current_segment = current_data[i:i + segment_length]
    vibration_segment = vibration_data[i:i + segment_length]
    labels, working = segment_signal(current_segment, previous_working=working)

    #如果检测到工作状态
    if 3 in labels:
        print(f"Detected working state at index {i}.")
        print("Accumulating vibration data...")

        #维护长度为10000的振动数据窗口，一旦超过10000就将前面的数据丢弃，确保后面加入的数据完整
        if accumulated_vibration.size + segment_length > 10000:
            overflow = accumulated_vibration.size + segment_length - 10000
            accumulated_vibration = np.concatenate((accumulated_vibration[overflow:], vibration_segment))
        else:
            accumulated_vibration = np.concatenate((accumulated_vibration, vibration_segment))

        #将10000的振动数据窗口传入模型进行测试
        if accumulated_vibration.size == 10000:
            loss = test(model, accumulated_vibration, device)
            #更新损失
            if loss is not None:
                print(f"Loss for segment at index {i}: {loss}")
                losses.append(loss)
            accumulated_vibration = np.array([], dtype=np.float32)
    else:
        print(f"No working state at index {i}.")
    
    return labels, working, accumulated_vibration

def record_times_and_write_csv(times, file_path):
    """
    记录时间并写入 CSV 文件。
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['迭代次数', '执行时间（秒）'])
        for index, time_value in enumerate(times):
            writer.writerow([index, time_value])

def plot_and_print_statistics(current_data, vibration_data, all_labels, losses, start_index, end_index, image_dir):
    """
    绘制图形和打印统计数据，并将图像保存在指定的目录。
    参数:
    current_data : np.array
        电流数据。
    vibration_data : np.array
        振动数据。
    all_labels : np.array
        所有段的标签。
    losses : list
        工作段的损失值列表。
    start_index : int
        绘图的起始索引。
    end_index : int
        绘图的结束索引。
    image_dir : str
        图像保存的目录。
    """

    # 绘制电流信号、振动信号和标签的图形
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.plot(current_data[start_index:end_index], label='Current Signal')
    plt.title('Waveform of All Segments')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(vibration_data[start_index:end_index], label='Vibration Signal')
    plt.title('Vibration Signal Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Vibration Signal Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.step(range(len(all_labels)), all_labels, label='Operation Labels', where='post')
    plt.title('Label Changes Over Segments')
    plt.xlabel('Segment Index')
    plt.ylabel('Label Value')
    plt.legend()

    plt.tight_layout()
    combined_fig_path = os.path.join(image_dir, "Combined_Waveform.png")
    plt.savefig(combined_fig_path)
    plt.close()

    if losses:
        average_loss = np.mean(losses)
        print(f"Average Loss: {average_loss}")
        plt.figure(figsize=(10, 6))
        plt.plot(losses, marker='o', linestyle='-')
        plt.title('Loss Over Working Segments')
        plt.xlabel('Segment Index')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_fig_path = os.path.join(image_dir, "Loss_Over_Time.png")
        plt.savefig(loss_fig_path)
        plt.close()
    else:
        print("No working segments found for loss calculation.")


def main_process(date, base_path, model_path, input_dim, hidden_dim, output_dim, device, segment_length=2000):
    # 构建文件路径
    current_file_path = os.path.join(base_path, 'data', 'Current', f'00001-20231024-0014_10_{date}.bin')
    vibration_file_path = os.path.join(base_path, 'data', 'Vibration', f'00001-20231024-0014_01_{date}.bin')
    csv_path = os.path.join(base_path, 'test_output', f'{date}_Runtime.csv')
    image_dir = os.path.join(base_path, 'test_output', 'images', date)
    
    # 确保所有必要的目录都存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    model = load_model(model_path, input_dim, hidden_dim, output_dim, device)
    current_data = np.fromfile(current_file_path, dtype=np.float32)
    vibration_data = np.fromfile(vibration_file_path, dtype=np.float32)

    '''
    #从头开始遍历整个数据集
    num_segments = len(current_data) // segment_length
    start_index = 0
    '''

    #从index 117000100 开始运行 1000次
    num_segments = 1000  
    start_index = 117000100
    end_index = start_index + num_segments * segment_length
    

    accumulated_vibration = np.array([], dtype=np.float32)
    all_labels = np.array([])
    losses = []
    working = False
    times = []

    for i in range(start_index, end_index, segment_length):
        start_time = time.time()
        labels, working, accumulated_vibration = process_segment(
            current_data, vibration_data, i, segment_length, model, device, working, accumulated_vibration, losses
        )
        end_time = time.time()
        times.append(end_time - start_time)
        all_labels = np.concatenate((all_labels, labels))

    record_times_and_write_csv(times,csv_path)
    plot_and_print_statistics(current_data, vibration_data, all_labels, losses, start_index, end_index, image_dir)

    # 计算总执行时间和平均执行时间
    total_time = sum(times)
    average_time = total_time / len(times) if times else 0
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Execution Time per Segment: {average_time:.5f} seconds")    

def main():
    date = '2024-01-03'  # 设置你想要处理的数据日期
    base_path = os.path.dirname(__file__)  # 获取脚本文件当前位置的路径
    
    # 模型参数配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = 1
    hidden_dim = 16
    output_dim = 4
    #直接从训练输出目录中加载模型
    model_path = os.path.join(base_path, 'train_output', 'model' , 'model.pth')
    
    # 执行主处理流程
    main_process(date, base_path, model_path, input_dim, hidden_dim, output_dim, device)
if __name__ == "__main__":
    main()
