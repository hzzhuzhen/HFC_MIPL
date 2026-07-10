#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
# 提供构建神经网络的各种模块和损失函数。
import torch.nn as nn
# 提供各种神经网络功能的函数接口（如激活函数、卷积操作等）。
import torch.nn.functional as F
# 设置计算设备：
# 如果有可用的 CUDA GPU，则使用 GPU 进行计算；否则使用 CPU。
# torch.device("cuda")：指定使用 GPU。
# torch.device("cpu")：指定使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 GatedAttentionLayerV 类
class GatedAttentionLayerV(nn.Module):
    '''
    论文对应公式2
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
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
        # 输出结果标签
        return out_tanh

# GatedAttentionLayerU 的类，一个简单的门控注意力机制层。
class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    应用sigmoid函数于输入特征和权重矩阵的点积加上偏置项的结果。这是对某个方程（方程2）中步骤的文字描述
    '''
    # __init__ 方法初始化对象，设定默认维度为512。super() 调用父类 nn.Module 的构造函数，确保基础功能被正确初始化
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

# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class GatedAttention(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(GatedAttention, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        self.L = 500
        self.D = 128
        self.K = 1
        # self.args 存储传入的参数对象，
        self.args = args
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
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A

    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss(self, A, prediction, target, args):
        '''
        Equation (8)
        '''
        # 创建一个与 target 形状相同的零张量 Y_candiate，并将其设备设置为 device（假设在其他地方定义）。
        Y_candiate = torch.zeros(target.shape).to(device)
        # 然后将 target 中大于0的元素对应位置设为1，标记候选标签。
        Y_candiate[target > 0] = 1
        # 使用 Y_candiate 对预测结果 prediction 进行掩码操作，只保留候选标签对应的预测值。
        prediction_mask = prediction * Y_candiate
        # 对掩码后的预测值进行归一化处理，确保每个样本的预测概率之和为1。【候选标签的预测值/预测组再各个类上的总数（分配到各个类别上repeat效果），最后(0,1)调换顺序后成[class,batch_size]】
        new_prediction = prediction_mask / prediction_mask.sum(dim=1).repeat(prediction_mask.size(1), 1).transpose(0, 1)
        # 计算两个熵：entropy_Y 表示预测结果与目标标签之间的交叉熵，entropy_A 表示注意力权重的熵。
        entropy_Y = - target * torch.log(prediction)
        entropy_A = - A * torch.log(A)
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss


    # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective(self, X, Y, args):
        '''
        calculate the full loss, weighted partial labels, and attention scores
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
        Y_prob, _  = self.forward(X, args)
        # 使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)

        # 返回预测的概率分布 Y_prob，用于模型评估或测试。
        return Y_prob
    # 定义 evaluate_objective 方法，用于模型测试。
    def evaluate_objectiveV2(self, X, Y, args):
        '''
        model testing
        '''
        # 调用 forward 方法，获取预测的概率分布 Y_prob。忽略返回的注意力权重 _
        Y_prob, A = self.forward(X, args)
        # 使用 torch.clamp 函数限制 Y_prob 的值在 [1e-5, 1 - 1e-5] 范围内，防止数值不稳定。
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # 使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)
        # 2. 损失计算 调用 full_loss 方法计算新的预测概率 new_prob 和总损失 loss。
        new_prob, loss = self.full_loss(A, Y_prob, Y, args)

        # 返回预测的概率分布 Y_prob，用于模型评估或测试。
        return loss, new_prob, A