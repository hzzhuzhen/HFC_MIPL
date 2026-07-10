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

# mix-loss:CrossEntropyLoss: 加权交叉熵损失函数。
from weight_loss import CrossEntropyLoss as CE

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
    
    # a. 深度流行学习-单次迭代
    def deepm_loss(args, X_train_iter, Y_train_pred_iter,Y_train_P_iter, Y_train_lp_iter):
	    # 清除梯度：optimizer.zero_grad()，防止梯度累积。
	    #optimizer.zero_grad()
        Y_train_pred_iter = torch.from_numpy(Y_train_pred_iter).float().detach().cuda()
        Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
        X_train_iter.requires_grad = False
        Y_train_pred_iter.requires_grad = False
        Y_train_lp_iter.requires_grad = False

        # 调整 Y_pred_iter 的形状
        Y_train_pred_iter = Y_train_pred_iter.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]

        # 计算损失：根据是否使用标签传播，选择不同的损失函数：
        # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
        # 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
        # 检查 Y_pred_iter 的类型
        #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
        if args.using_lp:
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)
        else:
            deepm_loss = F.binary_cross_entropy(Y_train_pred_iter, Y_train_P_iter)

        # # 反向传播：loss.backward()计算梯度。
        # loss.backward()
        # # 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
        # optimizer.step()
        return deepm_loss
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
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
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

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
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
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
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

# ResidualGatedAttention2 残差门控注意力版本2（简单模式，残差插入第二次提取特征到门控注意力之间）
#
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class ResidualGatedAttention2(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(ResidualGatedAttention2, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        # self.args 存储传入的参数对象，
        self.args = args
        #500
        self.L = self.args.L  
        self.D = 128
        self.K = 1

        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_V2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        self.linear_U2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
    def forward(self, X, args,isResidual = False):
        # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
        X = X.squeeze(0)
        # 对小数据量的数据做特殊处理
        if self.nr_fea < 100:
            H = self.feature_extractor_part1_small(X)
        else:
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
        if not isResidual:
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
        
        #A_raw = self.attention_weights(A_V * A_U)
         # 加入残差网络
        if isResidual:
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            b,c,h,w=H.shape #[46,50,4,4]
            A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
            A2_avg = torch.mean(A2_raw, dim=2) #b,num_class
            A2_max = torch.max(A2_raw, dim=2)[0] #b,num_class
            A2 = A2_avg+self.la*A2_max
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A2_V = self.att_layer_V(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2_U = self.att_layer_U(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2 = self.attention_weights(A2_V * A2_U)
            A = A2
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A
    
    # 定义 forward 方法，它接收两个参数：输入张量 X 和参数对象 args。
    def forward_bak(self, X, args):
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
    
    # a. 深度流行学习-单次迭代
    def deepm_loss(args, X_train_iter, Y_train_pred_iter,Y_train_P_iter, Y_train_lp_iter):
	    # 清除梯度：optimizer.zero_grad()，防止梯度累积。
	    #optimizer.zero_grad()
        Y_train_pred_iter = torch.from_numpy(Y_train_pred_iter).float().detach().cuda()
        Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
        X_train_iter.requires_grad = False
        Y_train_pred_iter.requires_grad = False
        Y_train_lp_iter.requires_grad = False

        # 调整 Y_pred_iter 的形状
        Y_train_pred_iter = Y_train_pred_iter.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]

        # 计算损失：根据是否使用标签传播，选择不同的损失函数：
        # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
        # 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
        # 检查 Y_pred_iter 的类型
        #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
        if args.using_lp:
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)
        else:
            deepm_loss = F.binary_cross_entropy(Y_train_pred_iter, Y_train_P_iter)

        # # 反向传播：loss.backward()计算梯度。
        # loss.backward()
        # # 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
        # optimizer.step()
        return deepm_loss
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
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
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

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
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
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
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

# ResidualGatedAttention2 残差门控注意力版本2（简单模式，残差插入第二次提取特征到门控注意力之间）
#
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class ResidualGatedAttention2Plus(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(ResidualGatedAttention2Plus, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        # self.args 存储传入的参数对象，
        self.args = args
        #500
        self.L = self.args.L  
        self.D = 128
        self.K = 1

        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
            nn.Linear(self.nr_fea, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_V2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        self.linear_U2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
    def forward(self, X, args,isResidual = False):
        # 对图片数据的处理（"MNIST_MIPL", "FMNIST_MIPL"）
        if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
            # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
            X = X.squeeze(0)
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
        else: # 对小数据量的数据做特殊处理（ # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets）
            X = X.float()
            X = X.view(-1, args.nr_fea)
            H = self.feature_extractor_part1_small(X)
            H2 = H
        if not isResidual:
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
        
        #A_raw = self.attention_weights(A_V * A_U)
         # 加入残差网络
        if isResidual:
            # todo 参数数据形状处理
            b,c,h,w=H.shape #[46,50,4,4]
            A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
            A2_avg = torch.mean(A2_raw, dim=2) #b,num_class
            A2_max = torch.max(A2_raw, dim=2)[0] #b,num_class
            A2 = A2_avg+self.la*A2_max
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A2_V = self.att_layer_V(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2_U = self.att_layer_U(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2 = self.attention_weights(A2_V * A2_U)
            A = A2
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # # Equation (4):（参考但不使用）
        # A = (A - A.mean()) / (torch.std(A.detach()) + 1e-8)
        # A = A / math.sqrt(self.L)
        # A = F.softmax(A / self.tau, dim=1)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A
    
    # 定义 forward 方法，它接收两个参数：输入张量 X 和参数对象 args。
    def forward_bak(self, X, args):
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
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
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
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
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

# 定义 GatedAttentionLayerV 类
class ResidualGatedMultiHeadAttentionLayerV(nn.Module):
    '''
    论文对应公式2
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
    '''
    # __init__ 方法用于初始化类的成员变量。
    def __init__(self, input_dim=500, num_heads=5, head_dim=500,dim=512,la =2.0):
        super(ResidualGatedMultiHeadAttentionLayerV, self).__init__()
        self.la = la
        self.input_dim = input_dim  # 输入特征维度 (500)
        self.num_heads = num_heads  # 注意力头的数量 (50)
        self.head_dim = head_dim    # 每个头的输出维度 (500)
        # self.dim = dim: 将输入维度保存为类的成员变量。
        # 线性变换层，用于将输入映射到多头空间
        self.linear_q = nn.Linear(input_dim, num_heads * head_dim)  # Query
        self.linear_k = nn.Linear(input_dim, num_heads * head_dim)  # Key
        self.linear_v = nn.Linear(input_dim, num_heads * head_dim)  # Value
        # 输出线性层，用于融合多头输出
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)

        #self.dim = dim
        # self.linear = nn.Linear(dim, 1): 创建一个线性层，将输入从 dim 维映射到 1 维。
        self.linear = nn.Linear(input_dim, 1)


    # 前向传播方法 forward
    # features: 输入特征张量，形状通常为 (batch_size, num_instances, feature_dim)。
    # W_V: 权重矩阵，形状为 (feature_dim, 1)。
    # b_V: 偏置项，形状为 (1,)。
    def forward(self, x, W_V, b_V):
        batch_size = x.size(0)  # 获取批次大小
        # 线性变换：将输入映射到多头空间
        #q = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        #k = self.linear_k(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        #v = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        # 计算注意力权重 (Scaled Dot-Product Attention)
        #scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [36, 50, 50]
        #attention_weights = F.softmax(scores, dim=-1)  # [36, 50, 50]
        # 加权求和
        #attended_values = torch.matmul(attention_weights, v)  # [36, 50, 500]
        # 将多头输出拼接并映射回原始维度
        # attended_values = attended_values.view(batch_size, self.num_heads * self.head_dim)  # [36, 50 * 500]
        # out = self.output_linear(attended_values)  # [36, 500]
        # 使用 PyTorch 的 F.linear 函数通过给定的权重 W_U 和偏置 b_U 对输入特征进行线性变换。
        #out = F.linear(features, W_U, b_U)
        # b,c,h,w=H.shape #[46,50,4,4]
        # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
        attended_values = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        attention_weights =0
        out_avg = torch.mean(attended_values, dim=1) #b,num_class
        out_max = torch.max(attended_values, dim=1)[0] #b,num_class
        out = out_avg+self.la*out_max # [36, 500]
        # # out = F.linear(features, W_V, b_V): 使用 F.linear 函数对 features 进行线性变换，即 W_V^T * features + b_V。
        out = F.linear(out, W_V, b_V)
        # 对结果做tanh激活
        out_tanh = torch.tanh(out)
        # 输出结果标签
        #return out_tanh
        return out_tanh, attention_weights

# GatedAttentionLayerU 的类，一个简单的门控注意力机制层。
class ResidualGatedMultiHeadAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    应用sigmoid函数于输入特征和权重矩阵的点积加上偏置项的结果。这是对某个方程（方程2）中步骤的文字描述
    '''
    # __init__ 方法初始化对象，设定默认维度为512。super() 调用父类 nn.Module 的构造函数，确保基础功能被正确初始化
    def __init__(self, input_dim=500, num_heads=5, head_dim=500,dim=512,la =2.0):
        super(ResidualGatedMultiHeadAttentionLayerU, self).__init__()
        self.la = la
        self.input_dim = input_dim  # 输入特征维度 (500)
        self.num_heads = num_heads  # 注意力头的数量 (50)
        self.head_dim = head_dim    # 每个头的输出维度 (500)
        # self.dim = dim: 将输入维度保存为类的成员变量。
        # 线性变换层，用于将输入映射到多头空间
        self.linear_q = nn.Linear(input_dim, num_heads * head_dim)  # Query
        self.linear_k = nn.Linear(input_dim, num_heads * head_dim)  # Key
        self.linear_v = nn.Linear(input_dim, num_heads * head_dim)  # Value
        # 输出线性层，用于融合多头输出
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)
        # 将传入的维度参数保存为实例变量 self.dim，以便在类的其他方法中使用。
        #self.dim = dim
        # 实例化一个线性层 self.linear，它接受维度大小为 dim 的输入，并输出大小为1的张量。这个层会自动管理其内部权重和偏置。
        self.linear = nn.Linear(input_dim, 1)

    # 定义 forward 方法，这是 PyTorch 中定义如何从前向后传递数据的方式。
    # 此方法接收三个参数：features（输入特征），W_U（权重），和 b_U（偏置）
    def forward(self, x, W_U, b_U):
        batch_size = x.size(0)  # 获取批次大小 [36, 500]
        # 线性变换：将输入映射到多头空间
        # q = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        # k = self.linear_k(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        # v = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        # # 计算注意力权重 (Scaled Dot-Product Attention)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [36, 50, 50]
        # attention_weights = F.softmax(scores, dim=-1)  # [36, 50, 50]
        # # 加权求和
        # attended_values = torch.matmul(attention_weights, v)  # [36, 50, 500]
        # 将多头输出拼接并映射回原始维度
        # attended_values = attended_values.view(batch_size, self.num_heads * self.head_dim)  # [36, 50 * 500]
        # out = self.output_linear(attended_values)  # [36, 500]
        # 使用 PyTorch 的 F.linear 函数通过给定的权重 W_U 和偏置 b_U 对输入特征进行线性变换。
        #out = F.linear(features, W_U, b_U)
        # b,c,h,w=H.shape #[46,50,4,4]
        # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
        attended_values = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        attention_weights = 0
        out_avg = torch.mean(attended_values, dim=1) #b,num_class
        out_max = torch.max(attended_values, dim=1)[0] #b,num_class
        out = out_avg+self.la*out_max # [36, 500]
        out = F.linear(out, W_U, b_U)

        # 应用 sigmoid 函数到上一步得到的输出 out 上，生成一个新的输出 out_sigmoid。Sigmoid 函数将数值压缩到(0,1)区间，适用于转换为概率值的任务。
        out_sigmoid = torch.sigmoid(out)
        # 返回经过 sigmoid 函数处理后的结果 out_sigmoid。这代表了门控注意力机制的最终输出。
        return out_sigmoid, attention_weights

# 残差门控注意版本3（残差插入到门控注意力中，通过线性简单增注意力头，再残差，再门控）
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class ResidualGatedAttention3(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(ResidualGatedAttention3, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        self.L = 500
        self.D = 128
        self.K = 1
        # self.args 存储传入的参数对象，
        self.args = args
        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_RV = ResidualGatedMultiHeadAttentionLayerV(self.L)
        self.att_layer_RU = ResidualGatedMultiHeadAttentionLayerU(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_RV = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_RU = nn.Linear(self.L * self.K, self.args.nr_class)
        # self.linear_RV2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        # self.linear_RU2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
    def forward(self, X, args,isResidual = True):
        # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
        X = X.squeeze(0)
        # 对小数据量的数据做特殊处理
        if self.nr_fea < 100:
            H = self.feature_extractor_part1_small(X)
        else:
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
        if not isResidual:
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
        
        #A_raw = self.attention_weights(A_V * A_U)
         # 加入残差网络
        if isResidual:
            H2 = H.view(-1, 50 * 4 * 4) # [36,800]
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2) # [36,500]
            # b,c,h,w=H.shape #[46,50,4,4]
            # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
            # A2_avg = torch.mean(A2_raw, dim=2) #b,num_class
            # A2_max = torch.max(A2_raw, dim=2)[0] #b,num_class
            # A2 = A2_avg+self.la*A2_max
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A2_V,A2_V_weight = self.att_layer_RV(H2, self.linear_RV.weight, self.linear_RV.bias)
            A2_U,A2_U_weight = self.att_layer_RU(H2, self.linear_RV.weight, self.linear_RV.bias)
            A2 = self.attention_weights(A2_V * A2_U)
            A = A2
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A
    
    # a. 深度流行学习-单次迭代
    def deepm_loss(args, X_train_iter, Y_train_pred_iter,Y_train_P_iter, Y_train_lp_iter):
	    # 清除梯度：optimizer.zero_grad()，防止梯度累积。
	    #optimizer.zero_grad()
        Y_train_pred_iter = torch.from_numpy(Y_train_pred_iter).float().detach().cuda()
        Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
        X_train_iter.requires_grad = False
        Y_train_pred_iter.requires_grad = False
        Y_train_lp_iter.requires_grad = False

        # 调整 Y_pred_iter 的形状
        Y_train_pred_iter = Y_train_pred_iter.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]

        # 计算损失：根据是否使用标签传播，选择不同的损失函数：
        # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
        # 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
        # 检查 Y_pred_iter 的类型
        #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
        if args.using_lp:
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)
        else:
            deepm_loss = F.binary_cross_entropy(Y_train_pred_iter, Y_train_P_iter)

        # # 反向传播：loss.backward()计算梯度。
        # loss.backward()
        # # 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
        # optimizer.step()
        return deepm_loss
    
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
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
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
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



# 残差门控注意版本4（残差插入到门控注意力中，通过标准多头注意力qkv，再残差，再门控）
# 定义 GatedAttentionLayerV 类
class ResidualGatedMultiHeadAttentionLayerV4(nn.Module):
    '''
    论文对应公式2
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
    '''
    # __init__ 方法用于初始化类的成员变量。
    def __init__(self, input_dim=500, num_heads=5, head_dim=500,dim=512,la =2.0):
        super(ResidualGatedMultiHeadAttentionLayerV4, self).__init__()
        self.la = la
        self.input_dim = input_dim  # 输入特征维度 (500)
        self.num_heads = num_heads  # 注意力头的数量 (50)
        self.head_dim = head_dim    # 每个头的输出维度 (500)
        # self.dim = dim: 将输入维度保存为类的成员变量。
        # 线性变换层，用于将输入映射到多头空间
        self.linear_q = nn.Linear(input_dim, num_heads * head_dim)  # Query
        self.linear_k = nn.Linear(input_dim, num_heads * head_dim)  # Key
        self.linear_v = nn.Linear(input_dim, num_heads * head_dim)  # Value
        # 输出线性层，用于融合多头输出
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)

        #self.dim = dim
        # self.linear = nn.Linear(dim, 1): 创建一个线性层，将输入从 dim 维映射到 1 维。
        self.linear = nn.Linear(input_dim, 1)


    # 前向传播方法 forward
    # features: 输入特征张量，形状通常为 (batch_size, num_instances, feature_dim)。
    # W_V: 权重矩阵，形状为 (feature_dim, 1)。
    # b_V: 偏置项，形状为 (1,)。
    def forward(self, x, W_V, b_V):
        batch_size = x.size(0)  # 获取批次大小
        # 线性变换：将输入映射到多头空间
        q = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        k = self.linear_k(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        v = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        #计算注意力权重 (Scaled Dot-Product Attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [36, 50, 50]
        attention_weights = F.softmax(scores, dim=-1)  # [36, 50, 50]
        #加权求和
        attended_values = torch.matmul(attention_weights, v)  # [36, 50, 500]
        # 将多头输出拼接并映射回原始维度
        # attended_values = attended_values.view(batch_size, self.num_heads * self.head_dim)  # [36, 50 * 500]
        # out = self.output_linear(attended_values)  # [36, 500]
        # 使用 PyTorch 的 F.linear 函数通过给定的权重 W_U 和偏置 b_U 对输入特征进行线性变换。
        #out = F.linear(features, W_U, b_U)
        # b,c,h,w=H.shape #[46,50,4,4]
        # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
        #attended_values = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        #attention_weights =0
        out_avg = torch.mean(attended_values, dim=1) #b,num_class
        out_max = torch.max(attended_values, dim=1)[0] #b,num_class
        out = out_avg+self.la*out_max # [36, 500]
        # # out = F.linear(features, W_V, b_V): 使用 F.linear 函数对 features 进行线性变换，即 W_V^T * features + b_V。
        out = F.linear(out, W_V, b_V)
        # 对结果做tanh激活
        out_tanh = torch.tanh(out)
        # 输出结果标签
        #return out_tanh
        return out_tanh, attention_weights

# GatedAttentionLayerU 的类，一个简单的门控注意力机制层。
class ResidualGatedMultiHeadAttentionLayerU4(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    应用sigmoid函数于输入特征和权重矩阵的点积加上偏置项的结果。这是对某个方程（方程2）中步骤的文字描述
    '''
    # __init__ 方法初始化对象，设定默认维度为512。super() 调用父类 nn.Module 的构造函数，确保基础功能被正确初始化
    def __init__(self, input_dim=500, num_heads=5, head_dim=500,dim=512,la =2.0):
        super(ResidualGatedMultiHeadAttentionLayerU4, self).__init__()
        self.la = la
        self.input_dim = input_dim  # 输入特征维度 (500)
        self.num_heads = num_heads  # 注意力头的数量 (50)
        self.head_dim = head_dim    # 每个头的输出维度 (500)
        # self.dim = dim: 将输入维度保存为类的成员变量。
        # 线性变换层，用于将输入映射到多头空间
        self.linear_q = nn.Linear(input_dim, num_heads * head_dim)  # Query
        self.linear_k = nn.Linear(input_dim, num_heads * head_dim)  # Key
        self.linear_v = nn.Linear(input_dim, num_heads * head_dim)  # Value
        # 输出线性层，用于融合多头输出
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)
        # 将传入的维度参数保存为实例变量 self.dim，以便在类的其他方法中使用。
        #self.dim = dim
        # 实例化一个线性层 self.linear，它接受维度大小为 dim 的输入，并输出大小为1的张量。这个层会自动管理其内部权重和偏置。
        self.linear = nn.Linear(input_dim, 1)

    # 定义 forward 方法，这是 PyTorch 中定义如何从前向后传递数据的方式。
    # 此方法接收三个参数：features（输入特征），W_U（权重），和 b_U（偏置）
    def forward(self, x, W_U, b_U):
        batch_size = x.size(0)  # 获取批次大小 [36, 500]
        # 线性变换：将输入映射到多头空间
        q = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        k = self.linear_k(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        v = self.linear_v(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        # 计算注意力权重 (Scaled Dot-Product Attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [36, 50, 50]
        attention_weights = F.softmax(scores, dim=-1)  # [36, 50, 50]
        # 加权求和
        attended_values = torch.matmul(attention_weights, v)  # [36, 50, 500]
        # 将多头输出拼接并映射回原始维度
        # attended_values = attended_values.view(batch_size, self.num_heads * self.head_dim)  # [36, 50 * 500]
        # out = self.output_linear(attended_values)  # [36, 500]
        # 使用 PyTorch 的 F.linear 函数通过给定的权重 W_U 和偏置 b_U 对输入特征进行线性变换。
        #out = F.linear(features, W_U, b_U)
        # b,c,h,w=H.shape #[46,50,4,4]
        # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
        #attended_values = self.linear_q(x).view(batch_size, self.num_heads, self.head_dim)  # [36, 50, 500]
        #attention_weights = 0
        out_avg = torch.mean(attended_values, dim=1) #b,num_class
        out_max = torch.max(attended_values, dim=1)[0] #b,num_class
        out = out_avg+self.la*out_max # [36, 500]
        out = F.linear(out, W_U, b_U)

        # 应用 sigmoid 函数到上一步得到的输出 out 上，生成一个新的输出 out_sigmoid。Sigmoid 函数将数值压缩到(0,1)区间，适用于转换为概率值的任务。
        out_sigmoid = torch.sigmoid(out)
        # 返回经过 sigmoid 函数处理后的结果 out_sigmoid。这代表了门控注意力机制的最终输出。
        return out_sigmoid, attention_weights

# 残差门控注意版本4（残差插入到门控注意力中，通过标准多头注意力qkv，再残差，再门控）
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class ResidualGatedAttention4(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(ResidualGatedAttention4, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        self.L = 500
        self.D = 128
        self.K = 1
        # self.args 存储传入的参数对象，
        self.args = args
        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_RV = ResidualGatedMultiHeadAttentionLayerV4(self.L)
        self.att_layer_RU = ResidualGatedMultiHeadAttentionLayerU4(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_RV = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_RU = nn.Linear(self.L * self.K, self.args.nr_class)
        # self.linear_RV2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        # self.linear_RU2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
    def forward(self, X, args,isResidual = True):
        # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
        X = X.squeeze(0)
        # 对小数据量的数据做特殊处理
        if self.nr_fea < 100:
            H = self.feature_extractor_part1_small(X)
        else:
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
        if not isResidual:
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
        
        #A_raw = self.attention_weights(A_V * A_U)
         # 加入残差网络
        if isResidual:
            H2 = H.view(-1, 50 * 4 * 4) # [36,800]
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2) # [36,500]
            # b,c,h,w=H.shape #[46,50,4,4]
            # A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
            # A2_avg = torch.mean(A2_raw, dim=2) #b,num_class
            # A2_max = torch.max(A2_raw, dim=2)[0] #b,num_class
            # A2 = A2_avg+self.la*A2_max
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A2_V,A2_V_weight = self.att_layer_RV(H2, self.linear_RV.weight, self.linear_RV.bias)
            A2_U,A2_U_weight = self.att_layer_RU(H2, self.linear_RV.weight, self.linear_RV.bias)
            A2 = self.attention_weights(A2_V * A2_U)
            A = A2
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A
    
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
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
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
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


# 偏标签单样本对比学习的注意力网络（原方法基础上加上原型对比学习损失）
# 加对比损失
# 定义偏标签场景下的原型对比损失
class PartialLabelContrastiveLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, temperature=0.1):
        super(PartialLabelContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        # 初始化原型向量
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, candidate_labels, predictions):
        """
        计算偏标签场景下的原型对比损失
        :param features: 样本特征 [batch_size, feature_dim]
        :param candidate_labels: 候选标签集合 [batch_size, num_classes] (one-hot 表示)
        :param predictions: 模型预测的概率分布 [batch_size, num_classes]
        :return: 对比损失
        """
        # 归一化特征和原型向量
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        batch_size, num_classes = candidate_labels.size()
        contrast_loss = 0.0

        for i in range(batch_size):
            # 获取当前样本的候选标签集合
            candidate_mask = candidate_labels[i]  # [num_classes]
            candidate_indices = torch.nonzero(candidate_mask).squeeze()  # 候选标签索引
            # 计算加权原型
            candidate_probs = predictions[i][candidate_indices]  # 候选标签的预测概率
            weighted_prototype = torch.sum(
                candidate_probs.view(-1, 1) * prototypes[candidate_indices], dim=0
            ) / torch.sum(candidate_probs)
            # 计算相似度矩阵
            sim_matrix = torch.matmul(features[i], prototypes.T) / self.temperature  # [num_classes]
            # 正类相似度
            positive_sim = torch.dot(features[i], weighted_prototype) / self.temperature
            # 对比损失
            exp_positive = torch.exp(positive_sim)
            exp_all = torch.exp(sim_matrix).sum()
            contrast_loss += -torch.log(exp_positive / exp_all)
        return contrast_loss / batch_size
# GatedAttention 的类，继承自 PyTorch 的 nn.Module 类。这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class DeepManifoldContrastGatedAttention(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(DeepManifoldContrastGatedAttention, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        self.L = 500
        self.D = 128
        self.K = 1
        # self.args 存储传入的参数对象，
        self.args = args
        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        #self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
        #self.linear_V2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        #self.linear_U2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
        # 原型对比损失
        self.prototype_loss = PartialLabelContrastiveLoss(
            num_classes=self.args.nr_class,
            feature_dim=self.L,
            temperature=0.1
        )

    # 定义 forward 方法，它接收两个参数：输入张量 X 和参数对象 args。
    def forward(self, X, args,isResidual = False):
        # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
        X = X.squeeze(0)
        # 对小数据量的数据做特殊处理
        if self.nr_fea < 100:
            H = self.feature_extractor_part1_small(X)
        else:
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
        if not isResidual:
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
        
        #A_raw = self.attention_weights(A_V * A_U)
         # 加入残差网络
        if isResidual:
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
            b,c,h,w=H.shape #[46,50,4,4]
            A2_raw=self.fc(H).flatten(2) #b,num_class,hxw
            A2_avg = torch.mean(A2_raw, dim=2) #b,num_class
            A2_max = torch.max(A2_raw, dim=2)[0] #b,num_class
            A2 = A2_avg+self.la*A2_max
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A2_V = self.att_layer_V(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2_U = self.att_layer_U(A2, self.linear_V2.weight, self.linear_V2.bias)
            A2 = self.attention_weights(A2_V * A2_U)
            A = A2
        # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
        A = torch.transpose(A, 1, 0)
        # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        A = torch.sigmoid(A)
        # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
        M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
        # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
        Y_prob = self.classifier(M)
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A , H2  # 返回预测概率、注意力权重和特征表示
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter,H2):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)
        # 原型对比损失
        contrast_target =target.reshape(1,-1)
        contrast_loss = self.prototype_loss(H2, contrast_target, prediction)
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss +args.mu2 * contrast_loss 
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
        '''
        calculate the full loss, weighted partial labels, and attention scores
        '''
        # 将目标标签 Y 重塑为一维张量，以便后续处理。
        Y = Y.reshape(-1)
        # 1. 模型计算 调用 forward 方法，获取预测的概率分布 Y_prob 和注意力权重 A。
        Y_prob, A, H2 = self.forward(X, args)
        # 使用 torch.clamp 函数限制 Y_prob 的值在 [1e-5, 1 - 1e-5] 范围内，防止数值不稳定。
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # 然后使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)
        # 2. 损失计算 调用 full_loss 方法计算新的预测概率 new_prob 和总损失 loss。
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter,H2)
        # 返回总损失 loss、新的预测概率 new_prob 和注意力权重 A。
        return loss, new_prob, A

    # 定义 evaluate_objective 方法，用于模型测试。
    def evaluate_objective(self, X, args):
        '''
        model testing
        '''
        # 调用 forward 方法，获取预测的概率分布 Y_prob。忽略返回的注意力权重 _
        Y_prob, A ,H2  = self.forward(X, args)
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

# LossGatedAttention 损失门控注意力版本（丢门）
# 定义AttentionLayer类(自定义的注意力机制)
# 创建一个注意力层，其中包含一个线性变换层，用于生成注意力权重。
class MixLossAttentionLayer(nn.Module):
    # 创建了一个继承自nn.Module的AttentionLayer类。
    # 在构造函数中初始化了一个线性层，用于将输入特征映射到一个标量（dim -> 1）。
    def __init__(self, dim=512):
        super( MixLossAttentionLayer, self).__init__() # 构造函数，在实例化该类时自动调用，调用父类（nn.Module）的构造函数，这是必须的操作。
        self.dim = dim # 设置特征维度，默认为 512。
        # # 创建一个线性变换层，将输入特征从 dim 维度映射到 1 维。
        # 这个线性层的作用是：对每个位置的特征打分，作为 attention score 的原始值。
        self.linear = nn.Linear(dim, 1) 

    # 前向传播方法
    # features: 输入特征张量，形状为 [N, dim]，表示 N 个样本，每个样本有 dim 维特征。
    # W_1: 自定义传入的权重矩阵，用于替代默认的 self.linear.weight
    # b_1: 自定义偏置项，用于替代默认的 self.linear.bias
    # flag: 控制是否启用注意力机制的标志（1 启用，非 1 不启用）
    def forward(self, features, W_1, b_1, isMixLoss):
        # 如果 flag == 1，启用注意力机制。
        if isMixLoss:
            # 使用传入的权重和偏置对特征进行线性变换。
                # 使用传入的 W_1 和 b_1 对 features 做线性变换。
                # 输出 out_c 是每个样本的 attention score（未归一化），形状为 [N, 1]
            out_c = F.linear(features, W_1, b_1)
            # alpha: 对结果进行softmax-like处理，确保所有值都是正数且加起来为1，以生成注意力权重alpha。
            # 减去最大值是为了数值稳定性，防止指数溢出。
            out = out_c - out_c.max()
            # 对每个元素取指数，转换为正值。
            out = out.exp()
            # 对每一行求和，得到每个样本的 attention score 总和。（好像结果没变，如果线性聚合的不是1，则会有不同效果）
            out = out.sum(1, keepdim=True)
            # 归一化操作：将每个样本的 attention score 除以总和，得到 softmax-like 的 attention 权重。
            # alpha 的形状为 [N, 1]，表示每个样本的注意力权重。
            alpha = out / out.sum(0)
            # alpha.expand_as(features) 将 alpha 扩展成与 features 相同的形状 [N, dim]
            # 然后乘以 N（features.size(0) 是 batch size N。），这一步可能是为了补偿前面的归一化，使得最终加权平均不改变尺度。
            alpha01 = features.size(0)*alpha.expand_as(features)
            # 利用alpha对原始特征进行加权求和得到上下文向量context。（element-wise 乘法。）
            # torch.mul(...)：对 features 和 alpha01 做 element-wise 乘法。
            # 得到加权后的特征图 context，形状与 features 相同 [N, dim]
            context = torch.mul(features, alpha01)
        else: # 如果flag != 1
            # 则直接返回原始特征作为上下文
            context = features
            # alpha 初始化为全零向量 [N, 1]，表示没有注意力权重
            alpha = torch.zeros(features.size(0),1)
                
        # A = torch.transpose(alpha, 1, 0)
        # # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
        # A = torch.sigmoid(A)
        # # 返回加权后的特征context，中间输出out_c，以及注意力权重alpha。
        return context, out_c, alpha

# LossGatedAttention 损失门控注意力版本（丢门）
# 结合 交叉注意力(实例和注意力后的线性共享参数)+Deepm+sharp Loss
# 这个类实现了一个包含特征提取、注意力机制和分类的神经网络模型
class LossGatedAttention(nn.Module):
    # 初始化函数接收一个参数 args，这通常是一个包含各种配置选项的对象。
    def __init__(self, args):
        super(LossGatedAttention, self).__init__()
        # 设定几个固定维度和其他属性：
        # L=500 可能代表线性层的输出大小，D=128 可能是中间层的维度，K=1 可能表示注意力头的数量。
        # self.args 存储传入的参数对象，
        self.args = args
        #500
        self.L = self.args.L  
        self.D = 128
        self.K = 1

        # self.nr_fea 是从 args 中获取的一个属性，可能表示输入特征数量。
        self.nr_fea = self.args.nr_fea
        # 残差网络
        self.la = args.la
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
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
            nn.Linear(self.nr_fea, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )
        #  新引入的 loss 注意力
        #  注意力机制层（AttentionLayer）
        # 实例化之前定义好的 AttentionLayer 类，传入特征维度 L=500
        # 该层会对所有样本进行 attention 权重计算，并加权求和得到上下文向量
        self.att_layer = MixLossAttentionLayer(self.L)
        # 最终分类层（linear）
        # 对 attention 处理后的特征做分类
        # 输入维度为 L*K（默认为 500×1 = 500）
        self.linear = nn.Linear(self.L*self.K, self.args.nr_class)
        self.criterion = torch.nn.CrossEntropyLoss(size_average=True)
        self.weight_criterion = CE(aggregate='mean')
        # 下面是传统的注意力
        # 创建两个注意力层实例 att_layer_V 和 att_layer_U，它们分别接受大小为 L 的输入。这些层可能是自定义的，用于实现特定的注意力机制。
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        # 定义两个线性层 linear_V 和 linear_U，用于将注意力机制处理后的特征映射到类别空间，输出大小为 nr_class。
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_V2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        self.linear_U2 = nn.Linear(self.args.nr_class, self.args.nr_class)
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
    def forward(self, X, args,isMixLoss = True):
        Y_prob = None
        A = None
        out_c = None 
        # 对图片数据的处理（"MNIST_MIPL", "FMNIST_MIPL"）
        if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
            # 使用 squeeze(0) 方法移除输入张量 X 在维度0上的单一维度值（如果存在），这通常用于处理单个样本时去除不必要的批量维度。
            X = X.squeeze(0)
            # 将输入张量 X 传入特征提取的第一部分 feature_extractor_part1，这是一个包含卷积层和池化层的序列，用于从输入中提取初步的特征表示。
            H = self.feature_extractor_part1(X)
            # 而 50 * 4 * 4 是根据前一层输出的尺寸确定的。这一步将多维特征图展平为一维向量，以便后续全连接层处理。
            H2 = H.view(-1, 50 * 4 * 4)
            # 将展平后的特征向量传入特征提取的第二部分 feature_extractor_part2，这个部分包含一个线性层、Dropout层和ReLU激活函数，进一步提取高层次特征。
            H2 = self.feature_extractor_part2(H2)
        else: # 对小数据量的数据做特殊处理（ # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets）
            X = X.float()
            X = X.view(-1, args.nr_fea)
            H = self.feature_extractor_part1_small(X)
            H2 = H
        if not isMixLoss:
            # 分别使用 att_layer_V 和 att_layer_U 计算注意力权重 A_V 和 A_U。这两个注意力层接受特征表示 H 以及线性层 linear_V 的权重和偏置作为输入。
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            # 将 A_V 和 A_U 相乘并传入 attention_weights 层，生成最终的注意力权重 A。这一步通过结合两种注意力机制来细化注意力权重。
            A = self.attention_weights(A_V * A_U)
            # 调用 torch.transpose 函数交换张量 A 的第1维和第0维，调整其形状以适应后续矩阵运算的要求。
            A = torch.transpose(A, 1, 0)
            # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
            A = torch.sigmoid(A)
            # # Equation (4):（参考但不使用）
            # A = (A - A.mean()) / (torch.std(A.detach()) + 1e-8)
            # A = A / math.sqrt(self.L)
            # A = F.softmax(A / self.tau, dim=1)
            # 使用矩阵乘法 (torch.mm) 将注意力权重 A 应用于特征表示 H，并通过除以注意力权重的总和进行归一化。这一步实现了加权平均特征表示的计算，如方程 (3) 所示。
            M = torch.mm(A, H2) / torch.sum(A)   # Equation (3)
            # 将加权平均特征表示 M 传入分类器 classifier，得到预测的概率分布 Y_prob。分类器包含一个线性层和Sigmoid激活函数，输出k个类别。
            Y_prob = self.classifier(M) 
            out_c = Y_prob      
         # 加入混合损失注意力
        if isMixLoss:
            # 调用 attention layer：
            # H: 特征向量 [B, L]
            # self.linear.weight, self.linear.bias: 自定义传入的 linear 参数
            # flag: 控制是否启用 attention
            # 返回：
            # out: 加权后的特征 [B, L]
            # out_c: attention score 的原始输出 [B, 1]
            # alpha: attention 权重 [B]
            out, out_c, alpha = self.att_layer(H2, self.linear.weight, self.linear.bias, isMixLoss)
            # 对加权后的特征取均值（注意：这里是所有样本的平均！）
            # 得到 [1, L] 的上下文向量，这一步有点特殊，可能是在模拟 多实例学习（MIL） 或 全局池化
            out = out.mean(0,keepdim=True)
            # 使用全连接层对上下文向量进行分类，输出 [1, 4]
            Y_prob = self.linear(out)
            A = torch.transpose(alpha, 1, 0)
            # 应用Sigmoid函数到注意力权重 A 上，将其值压缩到 (0, 1) 区间内，使其可以解释为概率值或重要性权重
            A = torch.sigmoid(A) #[1,batch_size] 前后统一
            #A = alpha
        # 返回预测概率 Y_prob 和注意力权重 A。这样不仅提供了分类结果，还允许用户查看模型在决策过程中关注的部分。
        return Y_prob, A ,out_c
       
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_mix_loss_deepm(self, A,out_c, prediction, target, args,Y_train_lp_iter, isMixLoss = True):
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

        # mix_loss 计算
        mix_loss = 0
        if isMixLoss:
            # loss_1: 分类损失（cross entropy）。
            # loss_1 = criterion(y, bag_label)
            # loss_2: 注意力加权的 instance-level 损失。
            # 构造 instance-level 的标签（所有 instance 使用bag label相同的值，并且每个实例都有1个）。[instance_size,1]
            #targetTensor = target.type(torch.LongTensor) # 转成 tensor
            target_max_value = torch.argmax(target).to(device) # 最大值
            #instance_label_matrix = torch.stack(target_max_value * A.size(0)) # 变成[batch_size,instace_labels]
            #instance_labels = torch.argmax(instance_label_matrix, dim=1).type(torch.LongTensor) # 取最大值  [batch_size]
            instance_labels = target_max_value * torch.squeeze(torch.ones(A.size(1),1)).type(torch.LongTensor).to(device) #[36,1]
            mix_loss = self.weight_criterion(out_c, instance_labels, weights=torch.squeeze(A))
        
        # 总损失是两者的加权和。
        #loss = loss_1+ 2.0*loss_2

        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.w_entropy_A * mix_loss+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss

    # 定义 full_loss 方法，接收注意力权重 A、预测结果 prediction、目标标签 target 和参数对象 args。此方法根据方程(8)计算总损失。
    def full_loss_deepm(self, A, prediction, target, args,Y_train_lp_iter):
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
        
        # 流行学习损失计算
        deepm_loss = 0
        if Y_train_lp_iter is not None:
            Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
            #X_train_iter.requires_grad = False
            #Y_train_pred_iter.requires_grad = False
            #Y_train_lp_iter.requires_grad = False
            # 调整 Y_pred_iter 的形状
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            # 计算损失：根据是否使用标签传播，选择不同的损失函数：
            # 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
            #print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
            deepm_loss = F.mse_loss(Y_train_pred_iter, Y_train_lp_iter)

        # 稀疏损失 sparsity loss in Equation (7):
        sp_loss = 0            
        idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
        prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
        sp_loss = torch.norm(prob_candidate, p=1, dim=0)

        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss + args.mu * sp_loss
        
        # 计算总损失，包括交叉熵损失和加权的注意力熵损失。args.w_entropy_A 是注意力熵损失的权重。()
        #loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+args.lambda_recon_rate *deepm_loss
        # 返回归一化后的预测结果 new_prediction 和总损失 loss。
        return new_prediction, loss
  # 定义 calculate_objective 方法，用于计算完整的损失、加权的部分标签和注意力分数。
    def calculate_objective_deepm(self, X, Y, args,Y_train_lp_iter):
        '''
        calculate the full loss, weighted partial labels, and attention scores
        '''
        isMixLoss = False
        # 将目标标签 Y 重塑为一维张量，以便后续处理。
        Y = Y.reshape(-1)
        # 1. 模型计算 调用 forward 方法，获取预测的概率分布 Y_prob 和注意力权重 A。
        Y_prob, A , out_c= self.forward(X, args,isMixLoss)
        # 使用 torch.clamp 函数限制 Y_prob 的值在 [1e-5, 1 - 1e-5] 范围内，防止数值不稳定。
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # 然后使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)
        # 2. 损失计算 调用 full_loss 方法计算新的预测概率 new_prob 和总损失 loss。
        #new_prob, loss = self.full_mix_loss_deepm(A, out_c, Y_prob, Y, args,Y_train_lp_iter,isMixLoss)
        new_prob, loss = self.full_loss_deepm(A, Y_prob, Y, args,Y_train_lp_iter)
        # 返回总损失 loss、新的预测概率 new_prob 和注意力权重 A。
        return loss, new_prob, A

    # 定义 evaluate_objective 方法，用于模型测试。
    def evaluate_objective(self, X, args):
        '''
        model testing
        '''
        # 调用 forward 方法，获取预测的概率分布 Y_prob。忽略返回的注意力权重 _
        Y_prob, _, _  = self.forward(X, args)
        # 使用 F.softmax 函数对 Y_prob 进行softmax处理，确保其为有效的概率分布。
        Y_prob = F.softmax(Y_prob, dim=1)

        # 返回预测的概率分布 Y_prob，用于模型评估或测试。
        return Y_prob
