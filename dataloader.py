#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np  # 导入了numpy用于数值计算。
import random  # 导入了random用于随机操作。
import torch  # 导入了torch用于深度学习框架支持。
import scipy.io as io  # 导入了scipy.io用于读取MAT文件。
import torch.utils.data as data_utils # 导入了torch.util.data中的data_utils用于数据集管理。

# 函数 to_categorical（可以处理多标签）
# y: 输入的标签列表（或数组），其中每个元素是一个整数，表示类别索引。
# nr_class: 类别的总数，即 one-hot 编码向量的长度。
def to_categorical(y, nr_class):
    '''
    函数用于生成 one-hot 标签 generate one-hot label
    '''
    # 初始化一个长度为 nr_class 的列表 y_list，所有元素初始化为 0。这个列表将被用来存储 one-hot 编码的结果， 成为y_list=[nr_class,1]矩阵。
    y_list = [0] * nr_class
    # 遍历输入的标签列表 y 中的每一个元素 i。
    for i in y:
        # 对于每个 i，将 y_list 列表中对应索引位置的值设置为 1。这意味着如果 i 是某个类别的索引，则该类别的位置在 one-hot 编码向量中会被标记为 1。
        y_list[i] = 1
    # 将 y_list 转换为一个 NumPy 数组，并将其赋值给变量 y_cate。这样做是为了便于后续的数值计算和处理。
    y_cate = np.array(y_list)
    # 返回 y_cate，即生成的 one-hot 编码向量。
    return y_cate

# load_data_mat 的函数。该函数用于从 .mat 文件中加载数据，并进行预处理
# 定义一个名为 load_data_mat 的函数，它接受四个参数：
# mat_path: 包含数据的 .mat 文件路径。
# nr_fea: 特征的数量。
# nr_class: 类别的数量。
# normalize: 是否对特征进行归一化，默认为 False。
def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):
    '''
    load the dataset in mat format
    '''
    # 1.加载数据
    # 加载 .mat 文件并提取数据
    # 使用 scipy.io.loadmat 函数从指定路径加载 .mat 文件，并将内容存储在 data_mat 变量中
    # 提取 data 字段中的数据，并将其赋值给变量 data。
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    
    # 检查原始数据是否包含NaN——tk
    print(f"原始数据形状: {data.shape}")
    if hasattr(data, 'dtype'):
        print(f"原始数据类型: {data.dtype}")
    
    # 检查data中是否包含NaN
    if hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.floating):
        if np.isnan(data).any():
            print(f"警告：原始数据中包含 {np.isnan(data).sum()} 个NaN值")
        else:
            print("原始数据中不包含NaN值")
    # 2.初始化变量
    # all_ins_fea: 存储所有实例的特征。
    all_ins_fea = []
    # all_ins_fea_tmp: 临时存储特征的数组。
    # np.empty((0, nr_fea)) 创建了一个形状为 (0, nr_fea) 的数组。
    # 这个数组的行数为 0，列数为 nr_class。
    # 由于行数为 0，这个数组实际上是一个空数组，没有实际的元素。
    all_ins_fea_tmp = np.empty((0, nr_fea))
    # ins_num: 每个包中的实例数量列表。
    # bag_idx_of_ins: 实例对应的包索引列表。
    ins_num, bag_idx_of_ins = [], []
    # bag_lab: 包的真实标签列表。
    # [dummy_ins_lab: 实例对应的真实标签]（与包标签相同）。
    bag_lab, dummy_ins_lab = [], []
    # partial_bag_lab: 包的部分标签列表。
    partial_bag_lab = []
    # [partial_dummy_ins_lab: 实例对应偏标签]
    partial_dummy_ins_lab = np.empty((0, nr_class))
    # partial_bag_lab_processed: 处理后的部分包标签。
    partial_dummy_ins_lab_processed = np.empty((0, nr_class))
    partial_bag_lab, partial_bag_lab_processed = np.empty((0, nr_class)), np.empty((0, nr_class))
    # bag_cnt: 包计数器，初始值为 1。
    bag_cnt = 1
    # 3.遍历每个包并处理数据
    # 遍历 data 中的每一个包（假设 data 是一个二维数组，每一行代表一个包）
    for i in range(data.shape[0]):
        # 3.1.【all_ins_fea=当前包的所有实例特征】将当前包的特征添加到 all_ins_fea 中。
        all_ins_fea = np.vstack((all_ins_fea_tmp, data[i, 0]))
        # 更新 all_ins_fea_tmp 以保存当前状态
        all_ins_fea_tmp = all_ins_fea
        # 获取当前包中实例的数量，并将其添加到 ins_num 列表中。(matlab的包文件，喜欢放在data[i]下的第一个位置)
        ins_num_tmp = data[i, 0].shape[0]
        ins_num.append(ins_num_tmp)
        # 生成当前包中每个实例对应的包索引(一个列表)，并将其添加到 bag_idx_of_ins 列表中。
        bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp
        bag_idx_of_ins = bag_idx_of_ins + bag_idx_of_ins_tmp
        # 增加包计数器 bag_cnt。
        bag_cnt += 1
        # 3.2.处理包中真实标签 the ground-truth labels of bags
        # [bag_lab=当前包的真实标签]，并将其转换为列表形式（减去 1 是为了适应零索引）【data[i, 2]放的是这个包的偏标签，转成python的从0开始计数】。
        bag_lab_tmp = list(data[i, 2].flatten() - 1)
        # 将这些标签添加到 bag_lab 列表中。
        bag_lab = bag_lab + bag_lab_tmp
        # [dummy_ins_lab=每个实例的正确标签]为当前包中的每个实例生成相同的标签列表，并将其添加到 dummy_ins_lab 列表中。
        dummy_ins_lab_tmp = [bag_lab_tmp] * ins_num_tmp
        dummy_ins_lab = dummy_ins_lab + dummy_ins_lab_tmp
        # 3.3.处理包中偏标签 the partial labels of bags
        # [partial_bag_lab=获取当前包的部分标签]，并将其转换为列表形式（减去 1 是为了适应零索引）
        partial_bag_lab_tmp = list(data[i, 1].flatten() - 1)
        # 使用 to_categorical 函数将部分标签转换为 one-hot 编码格式。
        partial_bag_lab_tmp = to_categorical(partial_bag_lab_tmp, nr_class)
        # 前面扩展了一个维度
        partial_bag_lab_tmp = np.expand_dims(partial_bag_lab_tmp, axis=0)
        # 垂直叠加到 partial_bag_lab 中。
        partial_bag_lab = np.vstack((partial_bag_lab, partial_bag_lab_tmp))
        # [partial_dummy_ins_lab=当前包中的每个实例生成相同的部分标签]，并将其添加到 partial_dummy_ins_lab 中。
        partial_dummy_ins_lab_tmp = partial_bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        partial_dummy_ins_lab = np.vstack((partial_dummy_ins_lab, partial_dummy_ins_lab_tmp))

    # 4.转换和处理数据
    # 将 bag_idx_of_ins 转换为 NumPy 数组，并增加一个新的维度。
    bag_idx_of_ins = np.array(bag_idx_of_ins)
    bag_idx_of_ins = np.expand_dims(bag_idx_of_ins, axis=1)
    # 将 bag_lab 和 dummy_ins_lab 转换为 NumPy 数组。
    bag_lab = np.array(bag_lab)
    dummy_ins_lab = np.array(dummy_ins_lab)
    # 将 实例级真实包标签dummy_ins_lab、实例级包id bag_idx_of_ins 和 实例的所有特征 all_ins_fea 水平堆叠在一起，形成一个新的特征矩阵 lab_inx_fea。
    lab_inx_fea = np.hstack((dummy_ins_lab, bag_idx_of_ins, all_ins_fea))
    # nr_partial_lab_per_ins=计算每个实例的部分标签总和：
    # np.sum(partial_dummy_ins_lab, 1)：对 partial_dummy_ins_lab 每一行求和，得到每个实例的部分标签总和。
    # np.expand_dims(..., axis=1)：将上述结果扩展为二维数组（增加一个维度），以便后续进行广播操作
    nr_partial_lab_per_ins = np.expand_dims(np.sum(partial_dummy_ins_lab, 1), axis=1)
    # 归一化每个实例的部分标签(实例上：总和为1,正标签零点几，负标签直接0)：
    # 将 partial_dummy_ins_lab 中的每个元素除以其对应的总和 nr_partial_lab_per_ins，从而实现归一化处理。这样可以确保每个实例的部分标签之和为1。
    partial_dummy_ins_lab_processed = partial_dummy_ins_lab / nr_partial_lab_per_ins
    # 计算每个包的部分标签总和：
    # np.sum(partial_bag_lab, 1)：对 partial_bag_lab 每一行求和，得到每个包的部分标签总和。
    # np.expand_dims(..., axis=1)：将上述结果扩展为二维数组（增加一个维度），以便后续进行广播操作。
    nr_partial_lab_per_bag = np.expand_dims(np.sum(partial_bag_lab, 1), axis=1)
    # partial_bag_lab_processed=归一化每个包的部分标签(包上：总和为1,正标签零点几，负标签直接0)：
    # [partial_bag_lab_processed=处理后的权重型偏标签(总和为1)]将 partial_bag_lab 中的每个元素除以其对应的总和 nr_partial_lab_per_bag，从而实现归一化处理。这样可以确保每个包的部分标签之和为1。
    partial_bag_lab_processed = partial_bag_lab / nr_partial_lab_per_bag

    # 检查特征数据是否包含NaN-tk
    if np.isnan(all_ins_fea).any():
        print(f"警告：特征数据中包含 {np.isnan(all_ins_fea).sum()} 个NaN值")
        # 处理NaN值：用0填充
        all_ins_fea = np.nan_to_num(all_ins_fea, nan=0.0, posinf=0.0, neginf=0.0)
        print("已用0填充NaN值")
    else:
        print("特征数据中不包含NaN值")
    
    # 归一化特征（如果需要）
    if normalize:
        # 计算 all_ins_fea 的均值 data_mean 和标准差 data_std。
        data_mean, data_std = np.mean(all_ins_fea, 0), np.std(all_ins_fea, 0)
        print("data_mean:-tk",data_mean)
        print("data_std:-tk",data_std)
        
        # # 分析标准差为0的特征-tk
        # zero_std_cols = np.where(data_std == 0)[0]
        # if len(zero_std_cols) > 0:
        #     print(f"发现 {len(zero_std_cols)} 个标准差为0的特征列: {zero_std_cols}")
        #     print("这些列的特征值:")
        #     for col in zero_std_cols[:5]:  # 只显示前5个
        #         unique_vals = np.unique(all_ins_fea[:, col])
        #         print(f"  列 {col}: 唯一值 = {unique_vals}, 均值 = {data_mean[col]}")
        
        # # 检查是否有全零列
        # zero_cols = np.where(np.all(all_ins_fea == 0, axis=0))[0]
        # if len(zero_cols) > 0:
        #     print(f"发现 {len(zero_cols)} 个全零特征列: {zero_cols}")
        
        # # 检查是否有常数列
        # constant_cols = []
        # for col in range(all_ins_fea.shape[1]):
        #     if len(np.unique(all_ins_fea[:, col])) == 1:
        #         constant_cols.append(col)
        # if constant_cols:
        #     print(f"发现 {len(constant_cols)} 个常数特征列: {constant_cols}")
            
            
        # 计算 all_ins_fea 的最小值 data_min 和最大值 data_max
        data_min, data_max = np.min(all_ins_fea, 0), np.max(all_ins_fea, 0)
        
        # 分析标准差为0的特征-tk
        zero_std_cols = np.where(data_std == 0)[0]
        if len(zero_std_cols) > 0:
            print(f"发现 {len(zero_std_cols)} 个标准差为0的特征列: {zero_std_cols[:10]}...")
            print(f"这些列的值都是: {all_ins_fea[0, zero_std_cols[0]]:.6f}")
            
            # 方法1：直接移除常数列（推荐）
            print(f"移除 {len(zero_std_cols)} 个常数列...")
            all_ins_fea = np.delete(all_ins_fea, zero_std_cols, axis=1)
            data_mean = np.delete(data_mean, zero_std_cols)
            data_std = np.delete(data_std, zero_std_cols)
            # # 更新特征维度
            # original_nr_fea = args.LL
            # args.L = all_ins_fea.shape[1]
            # print(f"特征维度从 {original_nr_fea} 更新为 {args.nr_fea}")
            
            # 方法2：如果不想移除，可以保持原始值（不进行归一化）
            # data_std[data_std == 0] = 1.0  # 将标准差为0的列设为1，避免除零
        
        # all_ins_fea归一化后=使用 (all_ins_fea - data_mean) / data_std 对特征进行标准化处理，并将结果赋值给 all_ins_fea。
        all_ins_fea_norm = (all_ins_fea - data_mean) / data_std
        all_ins_fea = all_ins_fea_norm
        
        # 检查归一化后是否产生NaN-tk
        if np.isnan(all_ins_fea).any():
            print(f"警告：归一化后产生 {np.isnan(all_ins_fea).sum()} 个NaN值")
            # 处理NaN值：用0填充
            all_ins_fea = np.nan_to_num(all_ins_fea, nan=0.0, posinf=0.0, neginf=0.0)
            print("已用0填充归一化后的NaN值")

    # 转换为 PyTorch 张量
    # 将所有处理后的数据转换为 PyTorch 张量：
    # 使用 torch.from_numpy() 函数将 NumPy 数组转换为 PyTorch 张量，以便于后续的深度学习模型训练。
    all_ins_fea = torch.from_numpy(all_ins_fea)
    bag_idx_of_ins = torch.from_numpy(bag_idx_of_ins)
    dummy_ins_lab = torch.from_numpy(dummy_ins_lab)
    bag_lab = torch.from_numpy(bag_lab)
    partial_bag_lab = torch.from_numpy(partial_bag_lab)
    partial_bag_lab_processed = torch.from_numpy(partial_bag_lab_processed)
    # 返回处理后的1.包特征、2.实例所在的包索引、3.实例真实标签、4包的真实标签、5包偏标签、6.归一后包标签。
    return all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed

# 加载索引文件的函数
def load_idx_mat(idx_file):
    '''
    load the index in mat format
    '''
    # 加载 .mat 文件：
    # 使用 scipy.io.loadmat 函数从指定路径加载 .mat 文件，并将内容存储在 idx 变量中。
    idx = io.loadmat(idx_file)
    # 提取训练集和测试集索引：
    # 从 idx 中提取出 trainIndex 和 testIndex 字段，并分别存储在 idx_tr_np 和 idx_te_np 变量中。
    idx_tr_np = idx['trainIndex']
    idx_te_np = idx['testIndex']
    # 将索引转换为列表：
    # 将 idx_tr_np 和 idx_te_np 转换为一维 NumPy 数组并展平，然后将其转换为 Python 列表。
    idx_tr = list(np.array(idx_tr_np).flatten())
    idx_te = list(np.array(idx_te_np).flatten())
    # 打乱顺序：
    # 使用 random.shuffle() 函数随机打乱 idx_tr 和 idx_te 列表中的元素顺序。
    random.shuffle(idx_tr)
    random.shuffle(idx_te)
    # 返回结果：
    # 返回打乱顺序后的训练集索引 idx_tr 和测试集索引 idx_te。
    return idx_tr, idx_te

# 自定义的 PyTorch 数据集类 MIPLDataloader，用于处理多实例学习（MIL）中的数据。
# 该类继承自 torch.util.data.Dataset，并实现了初始化、创建数据包、获取数据长度和获取单个数据项的方法。
class MIPLDataloader(data_utils.Dataset):
    # 初始化方法 __init__
    def __init__(self, all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr, idx_te,
                 args, seed=1, train=True, normalize=False):
        # 参数解析：
        # all_ins_fea: 所有包实例的特征。
        # bag_idx_of_ins: 每个实例所属的包索引。
        # dummy_ins_lab: 实例真实标签。
        # bag_lab: 包真实标签。
        # partial_bag_lab_processed: 包偏标签（经过处理，成为权重）。
        # idx_tr: 训练集索引列表。
        # idx_te: 测试集索引列表。
        # nr_fea: 特征数量。
        # seed: 随机种子，默认为1。
        # train: 是否是训练模式，默认为 True。
        # normalize: 是否进行归一化，默认为 False。
        self.all_ins_fea = all_ins_fea
        self.bag_idx_of_ins = bag_idx_of_ins
        self.dummy_ins_lab = dummy_ins_lab
        self.bag_lab = bag_lab
        self.partial_bag_lab_processed = partial_bag_lab_processed
        self.idx_tr = idx_tr
        self.idx_te = idx_te
        self.train = train
        self.args = args
        self.nr_fea = args.nr_fea
        self.normalize = normalize

        # 创建数据包：
        # 如果是训练模式 (self.train 为 True)，调用 _create_bags() 方法生成训练数据包及其相关标签，
        # 并存储在 self.train_bags_list, self.train_ins_lab_list, self.train_partial_bag_lab_list, 和 self.train_true_bag_lab_list 中。
        if self.train:
            self.train_bags_list, self.train_ins_lab_list, self.train_partial_bag_lab_list, \
            self.train_true_bag_lab_list = self._create_bags()
        # 否则，生成测试数据包及其相关标签，并存储在 self.test_bags_list, self.test_ins_lab_list,
        # self.test_partial_bag_lab_list, 和 self.test_true_bag_lab_list 中。
        else:
            self.test_bags_list, self.test_ins_lab_list, self.test_partial_bag_lab_list, \
            self.test_true_bag_lab_list = self._create_bags()

        # 定义全局manifold变量
        self.X_recon_train = None
        self.Y_recon_train = None
        self.X_recon_test = None
        self.Y_recon_test = None


    # 创建数据包方法 _create_bags


    # 提取包数据：
    # bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i: 获取属于当前包 i 的所有实例的布尔索引。
    # bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag): 压缩维度。
    # bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]: 提取属于当前包的所有实例的特征。
    # ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]: 提取属于当前包的所有实例的标签。
    # partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]: 提取当前包的部分标签。
    # partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0): 将部分标签扩展为二维数组。
    # true_bag_lab = self.bag_lab[i - 1]: 提取当前包的真实标签。
    # bag = bag.reshape(bag.shape[0], 1, 28, 28): 将包数据重塑为适合输入到卷积神经网络的形状（假设输入数据是图像）。
    # 添加到列表：
    # 将提取的数据分别添加到 bags_list, ins_lab_list, partial_bag_lab_list, 和 true_bag_lab_list 中。
    # 返回结果：
    # 返回包含所有包及其相关信息的四个列表。

    def _create_bags(self):
        # 初始化空列表：
        # bags_list: 存储每个包的数据。
        # ins_lab_list: 存储每个包中实例的标签。
        # partial_bag_lab_list: 存储每个包的部分标签。
        # true_bag_lab_list: 存储每个包的真实标签。
        bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list = [], [], [], []
        # 根据训练/测试模式处理数据：
        # 如果是训练模式，遍历 self.idx_tr 列表中的每个索引 i。
        # 如果是测试模式，遍历 self.idx_te 列表中的每个索引 i。
        # 提取包数据：
        # bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i: 获取属于当前包 i 的所有实例的布尔索引。
        # bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag): 压缩维度。
        # bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]: 提取属于当前包的所有实例的特征。
        # ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]: 提取属于当前包的所有实例的标签。
        # partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]: 提取当前包的部分标签。
        # partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0): 将部分标签扩展为二维数组。
        # true_bag_lab = self.bag_lab[i - 1]: 提取当前包的真实标签。
        # bag = bag.reshape(bag.shape[0], 1, 28, 28): 将包数据重塑为适合输入到卷积神经网络的形状（假设输入数据是图像）。
        # 添加到列表：
        # 将提取的数据分别添加到 bags_list, ins_lab_list, partial_bag_lab_list, 和 true_bag_lab_list 中。
        if self.train:
            for i in self.idx_tr:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                true_bag_lab = self.bag_lab[i - 1]
                if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
                    bag = bag.reshape(bag.shape[0], 1, 28, 28)
                else:
                    bag = bag.reshape(bag.shape[0], 1, 1, self.nr_fea)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)
        else:
            for i in self.idx_te:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                true_bag_lab = self.bag_lab[i - 1]
                if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
                    bag = bag.reshape(bag.shape[0], 1, 28, 28)
                else:
                    bag = bag.reshape(bag.shape[0], 1, 1, self.nr_fea)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)

        # 返回包数据(bag.shape[0], 1, 28, 28)、所有实例的标签、部分包标签、真实包标签。
        return bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list

    # 获取数据长度方法 __len__
    # 返回数据集的长度：
    # 如果是训练模式，返回训练数据集中实例标签的数量。
    # 如果是测试模式，返回测试数据集中实例标签的数量。
    def __len__(self):
        if self.train:
            return len(self.train_ins_lab_list)
        else:
            return len(self.test_ins_lab_list)

    # 获取单个数据项方法 __getitem__
    # 根据索引获取数据：
    # 如果是训练模式，从 self.train_bags_list, self.train_partial_bag_lab_list, 和 self.train_true_bag_lab_list 中获取对应索引的数据。
    # 如果是测试模式，从 self.test_bags_list, self.test_partial_bag_lab_list, 和 self.test_true_bag_lab_list 中获取对应索引的数据。
    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            partial_bag_label = self.train_partial_bag_lab_list[index]
            true_bag_label = self.train_true_bag_lab_list[index]
        else:
            bag = self.test_bags_list[index]
            partial_bag_label = self.test_partial_bag_lab_list[index]
            true_bag_label = self.test_true_bag_lab_list[index]

        # 返回包数据(bag.shape[0], 1, 28, 28)、部分包标签、真实包标签、索引。
        return bag, partial_bag_label, true_bag_label, index
