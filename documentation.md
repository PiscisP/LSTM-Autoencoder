# data.py #



这个文件用来对数据集进行预处理以进行训练。分别加载电流和震动数据集，处理完成后将工作状态的震动数据 按照模型要求的格式集存储到指定目录下。

输入地址为 data文件夹下的Current 和 Vibration， 输出地址为data文件夹 和下面的csv文件夹

目前只能一次处理一天的数据，更改日期就能更换文件地址。

处理完成后的文件格式为 (批次数量, 步长 = 10000, 特征 = 1) 

### segment_signal

对信号进行分段处理，根据阈值进行工况分割. 注意data这里使用的旧版的segment_signal算法，只能对较大的数据集进行工况分割，因为 `signal.find_peaks()`函数无法在小的数据段上找到好几个peak。所以在test.py 里用了简化版的segment_signal 以进行实时工况分析。



# train.py #

LSTM-Autoencoder 模型结构

LSTMAutoencoder(
  (encoder1): LSTM(1, 16, batch_first=True)				
  (encoder2): LSTM(16, 4, batch_first=True)					
  (decoder1): LSTM(4, 16, batch_first=True)
  (output): Linear(in_features=16, out_features=1, bias=True)
)



参数量计算公式

input_dim, hidden_dim, output_dim = 1, 16, 4

params_encoder1 = (input_dim + hidden_dim + 1) * 4 * hidden_dim

params_encoder2 = (hidden_dim + output_dim + 1) * 4 * output_dim

params_decoder1 = (output_dim + hidden_dim + 1) * 4 * hidden_dim

params_output = hidden_dim * input_dim + input_dim

参数量：

- **encoder1 (LSTM)**: 1152 个参数
- **encoder2 (LSTM)**: 336 个参数
- **decoder1 (LSTM)**: 1344 个参数
- **output (线性层)**: 17 个参数





这个文件用来训练模型， 通过一个文件列表将所有预处理过后的文件裁剪成相同大小的片段并加载进训练集和验证集。

输入地址为data文件夹，输出地址为train_output文件夹

目前采用了5天的工作数据进行训练，每天取工作数据前 `segment_length=1000` 个批次的80%存入训练集，20%存入验证集

训练集和验证集的shape为 

X_train.shape(4000, 10000, 1)
X_val.shape(1000, 10000, 1)



用num_epochs = 150, batch_size = 64, learning_rate = 0.001  的参数训练，loss收敛比较明显。

# test.py # 

test.py文件用于加载模型对测试数据进行测试。

注意这个文件的输入为原始的.bin文件，与train.py中的训练集不同。

输入地址为data中的Current和Vibration文件夹，输出地址为test_output文件夹

### segment_signal ###

test.py中的工况分割算法直接使用最大值进行判断。 将总长2000的数据平分成5段，每段取最大值进行平均，如果平均值超过设定的最大阈值（0.06），就算整段是工作状态。然后通过`previous_working`变量跟踪上一段的状态，如果上一段为工作状态，则采用继续工作状态阈值（0.045）略微低于最大阈值。同时对于异常装填直接设置了最大值（0.2），超过的直接标记为0，及异常状态。



### main_process

主要的逻辑是每一秒调用一次segment_signal 分析工况，如果检测为工作状态，添加入维护的10000长度的窗口中，将10000长度累积慢后传入模型进行预测，然后记录损失。对main_process的循环进行了计时。

最后在output文件夹中保存电流，震动和labels的图像，和记录每次循环时间的csv文件。打印出平均loss，平均时间和总时间。更改日期即可切换文件。



# 下一步

需要对大量的数据进行测试，以测试工作状态分割算法的鲁棒性， 以及是否能够过滤掉异常值带来的影响。

总结出一个规律来根据输出的loss判断是否发生故障，比如在短时间内突然变化，或者设定一个阈值。

也许可以通过标记 电流工作状态来训练另一个模型，从而更加精准地分割工作状态。

