import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

#加载数据
def load_data(file_paths, train_ratio=0.8, segment_length=1000):
    '''
    加载和处理来自多个文件的数据。函数首先将数据正规化并调整到指定的片段长度，然后将数据划分为训练集和验证集。
    参数：
    file_paths：包含数据文件路径的列表。
    train_ratio：用于训练的数据比例。
    segment_length：每个数据片段的长度。
    返回：
    训练和验证数据集。
    '''
    train_data = []
    val_data = []
    for file_path in file_paths:
        data = np.load(file_path)
        
        normalized_data = data['arr_0']
        print(normalized_data.shape) #打印每个文件的形状

        # 截取每个文件的前segment_length 个数据点
        if normalized_data.shape[0] > segment_length:
            normalized_data = normalized_data[:segment_length]

        # 根据比例划分训练和验证数据
        train_len = int(len(normalized_data) * train_ratio)
        #前80%的数据作为训练数据，后20%的数据作为验证数据
        train_data.append(normalized_data[:train_len])
        val_data.append(normalized_data[train_len:])

    X_train = np.concatenate(train_data)
    X_val = np.concatenate(val_data)

    print(f"X_train.shape{X_train.shape}")
    print(f"X_val.shape{X_val.shape}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)


    return train_dataset, val_dataset

#定义模型
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
    

#循环训练

def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    '''
    训练LSTM自编码器模型，并计算每个epoch的训练和验证损失。
    参数：
    model：训练的模型。
    train_loader：训练数据的加载器。
    valid_loader：验证数据的加载器。
    criterion：损失函数。
    optimizer：优化算法。
    device：训练设备（CPU或GPU）。
    num_epochs：训练的总轮数。
    返回：
    训练和验证过程中的损失值列表。
    '''
    train_losses = []
    valid_losses = []

    #循环训练
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)
        #打印每个循环的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    return train_losses, valid_losses

#保存模型和绘制损失曲线
def save_and_plot(model, train_losses, valid_losses, saving_dir, image_dir):
    
    # 确保保存目录存在
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    # 确保图片目录存在
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    #保存模型
    torch.save(model.state_dict(), os.path.join(saving_dir, 'model.pth'))
    
    #绘制训练损失曲线
    plt.figure()
    plt.semilogy(train_losses)
    plt.title("Training Loss")
    plt.savefig(os.path.join(image_dir, "Train_Loss.png"))
    plt.show()
    plt.close()
    
    
    #绘制验证损失曲线
    plt.figure()
    plt.semilogy(valid_losses)
    plt.title("Validation Loss")
    plt.savefig(os.path.join(image_dir, "Valid_Loss.png"))
    plt.show()
    plt.close()
    

def main():
    '''
    主函数，设置训练的参数，加载数据，创建模型，执行训练，并保存结果。
    
    '''
    # 获取脚本文件当前位置的路径
    base_path = os.path.dirname(__file__)

    # 构建数据文件的相对路径
    file_paths = [
        os.path.join(base_path, "data", "vibration_2023-12-31.npz"),
        os.path.join(base_path, "data", "vibration_2024-01-02.npz"),
        os.path.join(base_path, "data", "vibration_2024-01-03.npz"),
        os.path.join(base_path, "data", "vibration_2023-12-20.npz"),
        os.path.join(base_path, "data", "vibration_2023-12-26.npz")
    ]

    #定义参数
    input_dim = 1
    hidden_dim = 16
    output_dim = 4

    num_epochs = 150
    batch_size = 64
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #加载loader
    train_dataset, val_dataset = load_data(file_paths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #加载模型
    model = LSTMAutoencoder(input_dim, hidden_dim, output_dim).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #进行训练
    train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs)
    
    saving_dir = os.path.join(base_path, "train_output", "model")
    image_dir = os.path.join(base_path, "train_output", "pic")

    #保存模型和绘制损失曲线
    save_and_plot(model, train_losses, valid_losses, saving_dir, image_dir)

if __name__ == "__main__":
    main()
