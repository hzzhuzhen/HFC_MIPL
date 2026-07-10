#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 从 __future__ 模块导入 print_function，确保兼容 Python 2 和 Python 3 的 print 函数
from __future__ import print_function
import argparse  # 用于解析命令行参数
import os   # 提供与操作系统交互的功能
import time  # 提供时间相关的函数
import torch  # PyTorch 深度学习框架
import torch.utils.data as data_utils  # 提供数据加载和处理功能
import torch.optim as optim  # 提供优化算法
from torch.autograd import Variable  # 提供自动求导变量
from sklearn.metrics import accuracy_score  # 用于计算准确率

from dataloader import *  # 从自定义的 dataloader 模块中导入所有内容
from model import *  # 从自定义的 model 模块中导入所有内容
from utils import *  # 从自定义的 util 模块中导入所有内容

# 训练设置
# training settings
parser = argparse.ArgumentParser(
    description='Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning')
# 添加命令行参数
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training禁用 CUDA 训练')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)训练的 epoch 数量 (默认: 100)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)学习率 (默认: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R', help='weight decay权重衰减')
parser.add_argument('--w_entropy_A', type=float, default=0.0005, metavar='L', help='weight of the loss function损失函数的权重')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)随机种子 (默认: 123)')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path数据集路径')
parser.add_argument('--index', type=str, default='index', help='index path索引路径')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='MNIST_MIPL, FMNIST_MIPL, ...数据集名称 MNIST_MIPL, FMNIST_MIPL, ...')
parser.add_argument('--ds_suffix', type=str, default='1', help='the specific type of the data set数据集的具体类型')
parser.add_argument('--bs_tr', type=int, default=1, help='batch size for training 训练时的 batch size')
parser.add_argument('--bs_te', type=int, default=1, help='batch size for testing 测试时的 batch size')
parser.add_argument('--nr_fea', type=int, default=784, help='feature dimension of an instance 实例的特征维度')
parser.add_argument('--nr_class', type=int, default=5, help='classes of bag 包的类别数')
parser.add_argument('--normalize', type=str2bool, default=False, help='normalize the dataset, True or False 是否对数据集进行归一化，True 或 False')
# 解析命令行参数
args = parser.parse_args()

# 设置随机种子以确保结果可重复
seed_everything(args.seed)

# 判断是否使用 CUDA（GPU）
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 判断是否使用 CUDA（GPU）
if args.cuda:
    print('\nGPU is available!')

# 定义所有可用的数据集索引文件
all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
             'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']

#  evaluate 的函数， 这个函数的主要目的是对给定数据集进行模型评估，并计算模型的分类准确率。
# 将模型设置为评估模式。
# 遍历数据加载器中的每一个批次数据。
# 对每个批次数据进行前向传播，获取预测结果。
# 收集真实标签和预测标签，并将其转换为适合计算准确率的格式。
# 计算并返回最终的准确率。
def evaluate(loader, model):
    '''
    model testing
    '''
    # 将模型设置为评估模式。这一步非常重要，因为它会关闭 dropout 和 batch normalization 等训练时特有的层的行为，确保模型在测试时表现一致。
    model.eval()
    # 初始化变量
    # all_true_bag_lab：存储所有真实标签。
    all_true_bag_lab = []
    # all_pred_bag_lab：存储所有预测标签。
    all_pred_bag_lab = []
    # all_pred_bag_prob：存储所有预测概率，初始为空数组，形状为 (0, args.nr_class)，即没有样本且类别数为 args.nr_class。
    all_pred_bag_prob = np.empty((0, args.nr_class))
    # 遍历数据加载器
    # 使用 for 循环遍历数据加载器 loader。假设每个批次的数据包含四个元素：data（输入特征）、_（忽略的第一个元素）、true_bag_lab（真实标签）和 _（忽略的第二个元素）。
    for data, _, true_bag_lab, _ in loader:
        # 数据预处理
        # 将输入数据 data 和真实标签 true_bag_lab 移动到指定设备（CPU 或 GPU），并转换为 torch.float32 类型以确保数据类型一致。
        data = data.to(device)
        true_bag_lab = true_bag_lab.to(device)
        data = data.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # 模型推理
        # 调用模型的 evaluate_objective 方法进行推理，得到输出 output，这是一个包含预测概率的张量。
        output = model.evaluate_objective(data, args)
        # 处理预测结果
        # 将当前批次的预测概率 output 转换为 NumPy 数组并垂直堆叠到 all_pred_bag_prob 中，以便后续计算。
        all_pred_bag_prob = np.vstack((all_pred_bag_prob, output.detach().cpu().numpy()))
        # 使用 torch.max 函数找到每行中最大值的索引，即预测的类别标签。pred_bag_lab 是这些预测标签。
        _, pred_bag_lab = torch.max(output.data, 1)
        # 将当前批次的真实标签和预测标签分别添加到 all_true_bag_lab 和 all_pred_bag_lab 列表中。
        all_true_bag_lab.append(true_bag_lab.item())
        all_pred_bag_lab.append(pred_bag_lab.item())
    # 计算准确率
    # 将列表 all_true_bag_lab 和 all_pred_bag_lab 转换为 NumPy 数组，便于后续计算。
    all_true_bag_lab = np.array(all_true_bag_lab)
    all_pred_bag_lab = np.array(all_pred_bag_lab)
    # 使用 sklearn.metrics.accuracy_score 函数计算预测标签与真实标签之间的准确率。
    acc = accuracy_score(all_true_bag_lab, all_pred_bag_lab)
    # 返回计算得到的准确率 acc。
    return acc

# 函数的主要目的是在一个 epoch 内对训练数据进行模型训练，并计算和记录损失值。它通过以下步骤实现：
# 将模型设置为训练模式。
# 遍历训练数据加载器中的每一个批次数据。
# 对每个批次数据进行前向传播，获取预测结果和损失。
# 计算新的部分标签，并更新数据集中的部分标签列表。
# 执行反向传播和参数更新。
# 计算并返回整个 epoch 的平均训练损失。
def train(epoch):
    '''
     model training
     定义一个名为 train 的函数，接收一个参数 epoch（当前训练的 epoch 数）。该函数用于模型的训练。
    '''
    # 1.设置模型为训练模式并初始化变量
    # 将模型设置为训练模式，启用 dropout 和 batch normalization 等层。
    model.train()
    # 将模型设置为训练模式，启用 dropout 和 batch normalization 等层。
    train_loss = 0.
    # 初始化一个空数组用于存储注意力分数。
    attention_score_np = np.empty((0, 1))
    # 遍历训练数据加载器
    # 使用 for 循环遍历 train_loader 中的每一个批次数据。每个批次包含四个元素：
    # data：输入特征。
    # partial_bag_lab：部分标签（可能包含不完整的标签信息）。
    # true_bag_lab：真实标签。
    # index：样本索引。
    for batch_idx, (data, partial_bag_lab, true_bag_lab, index) in enumerate(train_loader):
        # 数据预处理
        # 如果启用了 CUDA（GPU），则将 data、partial_bag_lab 和 true_bag_lab 移动到 GPU 上。
        if args.cuda:
            data, partial_bag_lab, true_bag_lab = data.cuda(), partial_bag_lab.cuda(), true_bag_lab.cuda()
        # 将 data、partial_bag_lab 和 true_bag_lab 转换为 Variable，以便进行自动求导。
        data, partial_bag_lab, true_bag_lab = Variable(data), Variable(partial_bag_lab), Variable(true_bag_lab)
        # 将数据类型转换为 torch.float32，以确保数据类型一致。
        data = data.to(torch.float32)
        partial_bag_lab = partial_bag_lab.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # 重置梯度并计算损失 reset gradients
        # 清除优化器中累积的梯度，准备进行新的反向传播。
        optimizer.zero_grad()
        # 模型计算和损失计算核心逻辑。调用模型的 calculate_objective 方法，计算损失 loss、新的部分标签 new_partial_bag_lab 和注意力分数 attention_score。
        # calculate loss and metrics
        loss, new_partial_bag_lab, attention_score = model.calculate_objective(data, partial_bag_lab, args)
        # 将当前批次的损失累加到 train_loss 中。
        train_loss += loss.item()
        # 更新部分标签
        # 获取当前 epoch 对应的混合系数 lamda。
        lamda = lambda_list[epoch - 1]
        # 将 new_partial_bag_lab 和 partial_bag_lab 转换为 NumPy 数组。
        new_partial_bag_lab = new_partial_bag_lab.cpu().detach().numpy()
        partial_bag_lab = partial_bag_lab.cpu().detach().numpy()
        # manifold 对标签重构 形成新的特征标签 todo

        # 计算新的标签 new_label，它是原始部分标签和新预测的部分标签的加权和。
        new_label = lamda * partial_bag_lab + (1. - lamda) * new_partial_bag_lab
        # 压缩 new_label 的维度，使其符合预期形状。
        new_label = np.squeeze(new_label, axis=0)
        # 更新数据集中的部分标签列表 train_loader.dataset.train_partial_bag_lab_list。
        train_loader.dataset.train_partial_bag_lab_list[index] = new_label
        # 反向传播与参数更新 backward pass
        # 执行反向传播，计算梯度。
        loss.backward()
        # 使用优化器更新模型参数。
        optimizer.step()
    # 计算并打印平均损失 calculate loss and error for epoch
    # 计算整个 epoch 的平均损失。
    train_loss /= len(train_loader)
    # 如果是第一个 epoch 或者每 10 个 epoch，打印当前 epoch 的编号和训练损失。
    if epoch == 1 or (epoch) % 10 == 0:
        print('Epoch: {}, Train loss: {:.4f}.'.format(epoch, train_loss))
    # 返回训练后的模型对象和当前 epoch 的平均训练损失。
    return model, train_loss

# adjust_lambda 的函数，用于生成一个包含每个 epoch 的动量参数（lambda值）的列表。
# 该函数根据给定的总 epoch 数计算并返回一个 lambda 值列表，这些 lambda 值遵循特定的公式。
def adjust_lambda(epochs):
    '''
    the momentum parameter in Equation (7)
    $\lambda^{(t)} = \frac{T-t}{T}$
    定义一个名为 adjust_lambda 的函数，接收一个参数 epochs（总的训练 epoch 数）。
    该函数用于生成每个 epoch 的动量参数（lambda值），这些值遵循方程 (7)：
    '''
    # 初始化 lambda 列表
    # 创建一个长度为 epochs 的列表 lambda_list，初始时所有元素都设置为 1.0。这个列表将存储每个 epoch 对应的 lambda 值。
    lambda_list = [1.0] * epochs
    # 计算每个 epoch 的 lambda 值
    # 使用 for 循环遍历从 0 到 epochs-1 的每一个 epoch 索引 ep。
    # 对于每个 epoch ep，根据公式  λ^(t) =  (T−t)/T
    # 计算 lambda 值：
    # epochs - ep 表示当前 epoch 距离总 epoch 数的距离。
    # epochs 是总的 epoch 数。
    # (epochs - ep) / epochs 计算出当前 epoch 的 lambda 值，并将其存储在 lambda_list 中对应的位置。
    for ep in range(epochs):
        lambda_list[ep] = (epochs - ep) / (epochs)
    # 返回计算好的 lambda_list，它包含了每个 epoch 对应的 lambda 值。
    return lambda_list

# 主程序的入口，并实现了模型的训练和评估过程。它使用交叉验证（cross-validation）来评估模型性能，并记录每个试验和折叠（fold）的准确率。
# 初始化和加载数据集。
# 配置数据加载器和模型。
# 使用交叉验证进行模型训练和评估。
# 记录并输出每个试验和折叠的准确率，以及总的平均准确率和标准差。
# 记录并输出总运行时间。
# 通过这种方式，可以确保模型在不同的数据划分下都能得到有效的训练和评估，从而提高模型的泛化能力和可靠性。
if __name__ == "__main__":
    # 初始化时间和计算参数
    # 记录程序开始的时间 time_s，用于后续计算总运行时间。
    time_s = time.time()
    # 调用 adjust_lambda 函数生成一个包含每个 epoch 的 lambda 值的列表。
    lambda_list = adjust_lambda(args.epochs)
    # 设置试验次数 num_trial 为 1。
    num_trial = 1
    # 设置折叠数 num_fold 为 all_folds 列表的长度。
    num_fold = len(all_folds)
    # 数据加载器配置
    # 如果启用了 CUDA（GPU），则设置数据加载器的参数 loader_kwargs，包括 num_workers 和 pin_memory，否则为空字典。
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # 数据路径设置
    # 构建数据集路径 data_path、索引路径 index_path 和 .mat 文件路径 mat_path。
    # 提取数据集名称 ds_name，去掉文件扩展名。
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)
    mat_name = args.ds + '_r' + args.ds_suffix + '.mat'
    mat_path = os.path.join(data_path, mat_name)
    ds_name = mat_name[0:-4]
    # 加载数据
    # 调用 load_data_mat 函数加载数据集，并返回特征、实例标签、包标签等信息。
    all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
        mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

    # 初始化准确率数组
    # 初始化一个形状为 (num_trial, num_fold) 的空数组 accuracy，用于存储每次试验每个折叠的准确率。
    accuracy = np.empty((num_trial, num_fold))
    # 循环进行多次试验和交叉验证
    # 外层循环遍历每个试验 trial_i，内层循环遍历每个折叠 fold_i，并打印当前试验和折叠的信息。
    for trial_i in range(num_trial):
        for fold_i in range(num_fold):
            print('\n---------------- time: %d, fold: %d ----------------' % (trial_i + 1, fold_i + 1))
            idx_file = index_path + '/' + all_folds[fold_i]
            # 加载索引和数据集 load the index and dataset
            # 构建索引文件路径 idx_file 并加载训练和测试索引 idx_tr 和 idx_te。
            idx_tr, idx_te = load_idx_mat(idx_file)
            # 创建数据加载器
            # 使用 MIPLDataloader 创建训练和测试数据加载器 train_loader 和 test_loader，分别用于训练和测试阶段。
            train_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=True, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=True, **loader_kwargs)
            test_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=False, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=False, **loader_kwargs)

            # ---------------- 初始化模型和优化器 init model ----------------
            # 初始化模型 model，如果启用了 CUDA，则将模型移动到 GPU。
            model = GatedAttention(args)
            if args.cuda:
                model.cuda()
            # 初始化 SGD 优化器 optimizer，并设置学习率调度器 lr_scheduler。
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.reg)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # --------------开始训练 start training ---------------
            # 对每个 epoch 进行训练，调用 train 函数更新模型参数，并根据调度器调整学习率。
            for epoch in range(1, args.epochs + 1):
                model, loss = train(epoch)
                lr_scheduler.step()

            # --------------开始测试  start testing ---------------
            # 调用 evaluate 函数对测试集进行评估，获取测试准确率 test_accuracy。
            # 打印当前试验和折叠的测试准确率，并将其存储在 accuracy 数组中。
            test_accuracy = evaluate(test_loader, model)
            print('test_acc: {:.3f}'.format(test_accuracy))
            accuracy[trial_i, fold_i] = test_accuracy
    # 输出最终结果
    # 打印所有试验和折叠的平均准确率和标准差。
    print('The mean and std of accuracy at %d times %d folds: %f, %f' % (
    num_trial, num_fold, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))

    # 记录运行时间
    # 记录程序结束的时间 time_e，并计算总运行时间。
    # 打印总运行时间和训练完成的消息。
    time_e = time.time()
    print('\nRunning time is', time_e - time_s, 'seconds.')
    print('Training is finished.')
