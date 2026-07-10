#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import math
import torch
# 提供构建神经网络的各种模块和损失函数。
import numpy as np
import torch.nn as nn
# 提供各种神经网络功能的函数接口（如激活函数、卷积操作等）。
import torch.nn.functional as F
# 设置计算设备：
# 如果有可用的 CUDA GPU，则使用 GPU 进行计算；否则使用 CPU。
# torch.device("cuda")：指定使用 GPU。
# torch.device("cpu")：指定使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.(1) 辅助模型层-注意力V层
# 定义 GatedAttentionLayerV 类
class GatedAttentionLayerV(nn.Module):
    '''
    论文对应公式2
    $\text{tanh}\left(\boldsymbol{W}_{t}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_t \right)$ in Equation (2)
    '''
    # __init__ 方法用于初始化类的成员变量。
    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        # self.dim = dim: 将输入维度保存为类的成员变量。
        self.dim = dim
        # self.linear = nn.Linear(dim, 1): 创建一个线性层，将输入从 dim 维映射到 1 维。
        self.linear = nn.Linear(dim, 1)

    # 前向传播方法 forward
    # features: 输入特征张量，形状通常为 (batch_size, num_instances, feature_dim)。
    # W_V: 权重矩阵，形状为 (feature_dim, 1)。
    # b_V: 偏置项，形状为 (1,)。
    def forward(self, features, W_V, b_V):
        # out = F.linear(features, W_V, b_V): 使用 F.linear 函数对 features 进行线性变换，即 W_V^T * features + b_V。
        out = F.linear(features, W_V, b_V)
        # 对结果做tanh激活
        out_tanh = torch.tanh(out)

        return out_tanh

# 2.(2) 辅助模型层-注意力U层
# GatedAttentionLayerU 的类，一个简单的门控注意力机制层。
class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{s}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_s \right)$ in Equation (2)
    '''

    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        # 将传入的维度参数保存为实例变量 self.dim，以便在类的其他方法中使用。
        self.dim = dim
        # 实例化一个线性层 self.linear，它接受维度大小为 dim 的输入，并输出大小为1的张量。这个层会自动管理其内部权重和偏置。
        self.linear = nn.Linear(dim, 1)

    # 定义 forward 方法，这是 PyTorch 中定义如何从前向后传递数据的方式。
    # 此方法接收三个参数：features（输入特征），W_U（权重），和 b_U（偏置）
    def forward(self, features, W_U, b_U):
        # 使用 PyTorch 的 F.linear 函数通过给定的权重 W_U 和偏置 b_U 对输入特征进行线性变换。
        out = F.linear(features, W_U, b_U)
        # 应用 sigmoid 函数到上一步得到的输出 out 上，生成一个新的输出 out_sigmoid。Sigmoid 函数将数值压缩到(0,1)区间，适用于转换为概率值的任务。
        out_sigmoid = torch.sigmoid(out)
        # 返回经过 sigmoid 函数处理后的结果 out_sigmoid。这代表了门控注意力机制的最终输出。
        return out_sigmoid

# 1. 深度模型计算入口-门控注意力
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class GatedAttention(nn.Module):
    def __init__(self, args):
        # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        super(GatedAttention, self).__init__()
        # self.args 存储传入的参数对象，
        self.args = args
        self.L = self.args.L
        self.D = 128
        self.K = 1
        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea

        # 定义了第一个特征提取部分，包括两个卷积层（每个后面跟着ReLU激活函数和最大池化操作），用于从输入数据中提取初步特征。
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # 第二个特征提取部分，首先将之前提取的特征展平并通过一个全连接层转换为更高层次的特征表示，并应用Dropout防止过拟合。
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )
        # 第三个特征提取部分，针对小样本数据直接简单做特征提取
        self.feature_extractor_part1_small = nn.Sequential(
            nn.Linear(self.nr_fea, 50 * 4 * 4),
            nn.Dropout(),
            nn.ReLU(),
        )
        # Equation (2):
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        # 定义一个序列，用于生成注意力权重。首先通过一个线性层将类别数转换为中间维度 D，
        # 然后通过ReLU激活，最后再通过一个线性层将其转换为注意力头的数量 K。
        self.attention_weights = nn.Sequential(
            nn.Linear(self.args.nr_class, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),
        )
        # 最后定义一个分类器，将经过注意力机制处理后的特征（大小为 L * K）通过一个线性层映射到类别空间，
        # 并应用Sigmoid激活函数以获得最终的概率分布。这一步主要用于二分类或多标签分类任务。
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.args.nr_class),
            nn.Sigmoid()
        )
    # 定义 forward 方法，它接收两个参数：输入张量 X 和参数对象 args。
    def forward(self, X, args):
        # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
        X = X.squeeze(0)
        # todo 对小数据量的数据做特殊处理
        if self.nr_fea < 100:
            H = self.feature_extractor_part1_small(X)
        else:
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
        # 使用 view 方法重塑特征张量 H 的形状。这里的 -1 表示自动计算该维度的大小，
        # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
        H = H.view(-1, 50 * 4 * 4)
        # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
        H = self.feature_extractor_part2(H)
        # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
        A_V = self.att_layer_V(H, self.linear_V.weight, self.linear_V.bias)
        A_U = self.att_layer_U(H, self.linear_V.weight, self.linear_V.bias)
        # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
        A = self.attention_weights(A_V * A_U)
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # Equation (3):
        A = A / math.sqrt(self.L)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)  # Equation (4)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        Y_prob = self.classifier(Z)

        return Y_prob, A


    # # a. 深度流行学习-单次迭代
    # def deepm_loss(args, model_deepm, X_train_iter, Y_train_P_iter, Y_train_lp_iter):
    #     # 清除梯度：optimizer.zero_grad()，防止梯度累积。
    #     #optimizer.zero_grad()

    #     # # 计算索引位置和批次内的索引序号
    #     # start_idx = (i * batch_size) % X_train.shape[0]
    #     # idx = train_indicies[start_idx: start_idx + batch_size]
    #     # idx = train_indices # without batch

    #     # 获取当前批次的数据：根据索引 idx 获取当前批次的特征 X 和标签 Y_P 或 Y_lp。
    #     # get minibatch indices
    #     # randomly select a mini-batch of data
    #     # X = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
    #     # Y_P = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
    #     # Y_lp = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
    #     #X_train_iter = torch.from_numpy(X_train_iter).float().detach().cuda()
    #     Y_train_P_iter = torch.from_numpy(Y_train_P_iter).float().detach().cuda()
    #     Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
    #     X_train_iter.requires_grad = False
    #     Y_train_P_iter.requires_grad = False
    #     Y_train_lp_iter.requires_grad = False

    #     # 前向传播：将特征 X 输入模型，获取预测结果 Y_pred。
    #     Y_pred_iter,A = model_deepm.forward(X_train_iter,args)
    #     # 调整 Y_pred_iter 的形状
    #     Y_pred_iter = Y_pred_iter.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]

    #     # 计算损失：根据是否使用标签传播，选择不同的损失函数：
    #     # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
    #     # 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
    #     # 检查 Y_pred_iter 的类型
    #     #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
    #     if args.using_lp:
    #         deepm_loss = F.mse_loss(Y_pred_iter, Y_train_lp_iter)
    #     else:
    #         deepm_loss = F.binary_cross_entropy(Y_pred_iter, Y_train_P_iter)

    #     # # 反向传播：loss.backward()计算梯度。
    #     # loss.backward()
    #     # # 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
    #     # optimizer.step()
    #     return deepm_loss



    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss(self, A, prediction, target, args):
        '''
        the total loss function in Equation (9)
        '''
        # mapping loss in Equation (5):
        # 创建一个与 target 形状相同的零张量 Y_candiate，并将其设备设置为 device（假设在其他地方定义）。
        Y_candiate = torch.zeros(target.shape).to(device)
        # 然后将 target 中大于0的元素对应位置设为1，标记候选标签。
        Y_candiate[target > 0] = 1
        # 使用 Y_candiate 对预测结果 prediction 进行掩码操作，只保留候选标签对应的预测值。
        prediction_mask = prediction * Y_candiate
        # 对掩码后的预测值进行归一化处理，确保每个样本的预测概率之和为1。【候选标签的预测值/预测组再各个类上的总数（分配到各个类别上repeat效果），最后(0,1)调换顺序后成[class,batch_size]】
        new_prediction = prediction_mask / prediction_mask.sum(dim=1).repeat(prediction_mask.size(1), 1).transpose(0, 1)
        mp_loss = - target * torch.log(prediction)
        entropy_A = - A * torch.log(A)
        attention_loss = (torch.sum(entropy_A)) / entropy_A.size(0)

        # sparsity loss in Equation (7):
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)
        

        # inhibition loss in Equation (8):
        Y_non_candiate = torch.ones(target.shape).to(device) - Y_candiate
        non_prediction_mask = prediction * Y_non_candiate
        neg_prediction = (torch.ones(target.shape).to(device) - non_prediction_mask) * Y_non_candiate
        neg_prediction += Y_candiate
        in_loss = - Y_non_candiate * torch.log(neg_prediction)

        loss = torch.sum(mp_loss)+ args.w_entropy_A * attention_loss + args.mu * sp_loss + args.gamma * torch.sum(in_loss)  # Equation (9)
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

    # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective(self, X, Y, args):
        '''
        calculate the full loss
        '''
        # 将目标标签 Y 重塑为一维张量，以便后续处理。
        Y = Y.reshape(-1)
        # 1. 模型计算 调用 forward 方法，获取预测的概率分布 Y_prob 和注意力权重 A。
        Y_prob, A = self.forward(X, args)
        # 使用 torch.clamp 函数限制 Y_prob 的值在 [1e-5, 1 - 1e-5] 范围内，防止数值不稳定。
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # 然后使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)
        # 2. 损失计算 调用 full_loss 方法计算新的预测概率 new_prob 和总损失 loss。
        new_prob, loss = self.full_loss(A, Y_prob, Y, args)
        # 返回总损失 loss、新的预测概率 new_prob 和注意力权重 A。
        return loss, new_prob, A

    # 定义 evaluate_objective 方法，用于模型测试。
    def evaluate_objective(self, X, args):
        '''
        model testing
        '''
        # 调用 forward 方法，获取预测的概率分布 Y_prob。忽略返回的注意力权重 _
        Y_prob, _ = self.forward(X, args)
        # 使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)

        # 返回预测的概率分布 Y_prob，用于模型评估或测试。
        return Y_prob
