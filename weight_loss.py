#  PyTorch 实现的加权交叉熵损失函数
# 导入 PyTorch 核心库：
# torch：张量操作与自动求导。
# nn.Module：构建神经网络模块的基础类。
import torch
import torch.nn as nn

#  第二部分：log_sum_exp 函数（数值稳定性技巧，一种稳定的分类得分输出值的函数）
# 这个函数的目的是计算每个样本所有类别得分的 log-sum-exp，为了避免直接对 exp(x) 计算造成数值不稳定，这里使用了经典的 log-sum-exp 技巧
# 核心逻辑：其中 b=max(x)，即每行的最大值，作为偏移项，防止指数爆炸； b 是每行的最大值，作为偏移项；
# 将 x - b.repeat(...) 做减法后取指数，避免数值不稳定； 最后再把 b 加回来。

# 输入输出：
# 输入x: 一个形状为 [N, C] 的张量，表示 N 个样本，每个样本有 C 个类别的 logits。
# 输出: 一个形状为 [N] 的一维张量，每个元素是该样本所有类别得分的 log-sum-exp 值。
def log_sum_exp(x):
    # 注释部分引用了关于数值稳定技巧的参考链接，并提到 b 是偏移因子，用来提升数值稳定性。 See implementation detail in
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # b is a shift factor. see link.
    # x.size() = [N, C]:
    # import pdb; pdb.set_trace()
    # 使用 torch.max(x, 1) 获取每一行的最大值（即各个类别的最大值）。b 的形状是 [N]，每个元素是对应样本再各类别上的最大值。
    b, _ = torch.max(x, 1)
    # torch.squeeze(b)：去除所有大小为 1 的维度。但因为 b 本来就是 [N]，所以这个操作实际上没有效果。
    # torch.unsqueeze(..., dim=1)：在第 1 维度插入一个新维度，使 b 形状变为 [N, 1]。
    b = torch.unsqueeze(torch.squeeze(b),dim=1)
    # 核心模块：
    # 1. b.repeat(1, x.size(1)) 将 b（形状 [N, 1]）扩展成 [N, C]，与 x 同形状。
    # 2. x - b.repeat(...) 对 x 中每个元素减去该行最大值 b，以提高数值稳定性。
    # 3. torch.exp(...)对所有元素取指数。
    # 4. .sum(1, True) 在类别维度（dim=1）求和，结果形状为 [N, 1]。
    # 5. torch.log(...) 对 sum 结果取自然对数。
    # 6. + b 最后加上原来的偏移项 b，还原回原始 scale。【数值稳定性的一个损失】
    y = b + torch.log(torch.exp(x - b.repeat(1,x.size(1))).sum(1,True))
    # 将结果从 [N, 1] 转换为 [N]，返回给调用者。原本 y.size() = [N, 1].  Squeeze to [N] and return
    return y.squeeze(1)

# 第三部分：class_select 函数（选择目标类别的得分）
# 功能说明：
# 从 logits 中提取每个样本对应的真实类别得分（即 logits[i][target[i]]）。
# .eq(...) 比较构造出 one-hot 形式的掩码。
# 使用 masked_select 提取出对应的得分。
# 输入：
# logits: 模型输出的未归一化得分（logits），形状为 [N, C]，其中： N 是 batch size（样本数）， C 是类别数
# target: 真实标签，形状为 [N] 或 [N, 1]，表示每个样本的类别索引（0 到 C-1）
# 输出：一个形状为 [N] 的一维张量，表示每个样本在其真实类别上的输出得分。
def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    # 获取 batch size 和类别数。
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        # 构造一个 one-hot mask 矩阵（布尔型），标记哪些位置是目标类别。
        # torch.arange(0, num_classes)： 创建类别索引。生成从 0 到 num_classes - 1 的一维张量：
        # repeat(batch_size, 1)： 将上述一维张量复制 batch_size 次，形成一个二维张量：
        # .cuda(device)将这个张量移到指定的 GPU 上。[36,5]
        # target.data.repeat(num_classes, 1).t() ：把 target 标签复制 num_classes 次，并转置成 [N, C] 形状。
        # .eq(...) 比较两个 [N, C] 张量，得到一个布尔类型的 mask，其中只有目标类别的位置为 True。
        # 包装成 PyTorch Variable（适用于旧版本 PyTorch）。新版中可以省略
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    # 提取目标类别的得分
    # 使用 masked_select() 从 logits 中选出所有 True 位置的元素。
    # 因为 one_hot_mask 中每行只有一个 True（对应的目标类别），所以结果是一个长度为 N 的向量
    return logits.masked_select(one_hot_mask)

#  第四部分：cross_entropy_with_weights（带权重的交叉熵损失）
# logits: 模型输出的未归一化得分（logits），形状为 [N, C]，其中 N 是样本数量，C 是类别数量。
# target: 类别标签，[N] 或 [N, 1]，如果 target 是二维的，则压缩成一维。
# weights: 可选参数，形状为 [N]，用于对每个样本的损失进行加权。

# 计算交叉熵损失的核心部分：
# 交叉熵公式为：
# 所以这部分就是：
# log_sum_exp(logits)：
# class_select(...) ，即正确类别的得分
# 相减得到负对数似然损失（negative log likelihood loss）
def cross_entropy_with_weights(logits, target, weights=None):
    # 确保 logits 是一个二维张量，表示有 N 个样本，每个样本有 C 个类别的得分。    
    assert logits.dim() == 2
    # 确保 target 不需要计算梯度。这是因为 target 通常是标签，不需要对其进行梯度更新。
    assert not target.requires_grad
    # 如果 target 是二维的（即形状为 [N, 1]），则通过 .squeeze(1) 将其压缩成一维（即形状变为 [N]）。
    target = target.squeeze(1) if target.dim() == 2 else target
    # 确保处理后的 target 是一维的，即形状为 [N]。
    assert target.dim() == 1
    # import pdb; pdb.set_trace()

    # 计算损失(各项损失-选对的损失)：
    # log_sum_exp(logits)：计算所有类别的 log-sum-exp 值，这实际上是 softmax 的分母部分。
    # class_select(logits, target)：从 logits 中选择正确类别的得分。
    # 相减得到负对数似然损失（negative log likelihood loss）。【所有的损失-最大选择类别对应的损失】
    loss = log_sum_exp(logits) - class_select(logits, target)

    # 加入权重：
    # 若提供了 weights（shape 为 [N]），则对每个样本的 loss 进行加权。
    # 即：loss[i] *= weights[i]
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # 损失乘以注意力权重，Weight the loss [每个用例的权重不同]
        loss = loss * weights
    # 返回最终的 loss 向量,每个元素对应一个样本的损失值。
    return loss

# 第五部分：自定义的WeightCrossEntropyLoss 类（可配置聚合方式）
class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    # 初始化参数：
    # aggregate: 指定如何聚合 batch 内的 loss。
    # 'sum': 所有样本 loss 相加。
    # 'mean': 取平均。
    # None: 返回每个样本的 loss 向量。
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    # 前向传播逻辑：
    # 调用之前定义的 cross_entropy_with_weights 函数。
    # 根据 self.aggregate 对结果进行聚合或保持原样返回。
    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)

# 总结
# 函数	功能
# log_sum_exp(x)	数值稳定的 log(sum(exp(x)))，用于交叉熵计算
# class_select(logits, target)	提取每个样本的目标类别得分
# cross_entropy_with_weights(...)	计算每个样本的带权重交叉熵损失
# CrossEntropyLoss(...)	模块化封装，支持不同聚合方式
# 示例用法
# criterion = CrossEntropyLoss(aggregate='mean')
# logits = torch.randn(16, 10)  # batch_size=16, classes=10
# targets = torch.randint(0, 10, (16,))
# weights = torch.rand(16)
#
# loss = criterion(logits, targets, weights)
# loss.backward()