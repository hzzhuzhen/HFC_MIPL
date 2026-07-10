# numpy：用于数值计算和数组操作。虽然在这个函数中没有直接使用 numpy，但它可能是为了后续可能的扩展或与其他部分代码的兼容性而导入的。
# sklearn.metrics.f1_score：这是 Scikit-learn 库中的一个函数，用于计算 F1 分数。F1 分数是精确率（Precision）和召回率（Recall）的调和平均值，适用于二分类和多分类任务。在多标签分类任务中，f1_score 可以根据不同的平均策略计算多种类型的 F1 分数。
import numpy as np
from sklearn.metrics import f1_score

# 定义 metric 函数
# 参数：
# y：真实标签，形状为 (n_samples, n_labels)，表示 n_samples 个样本，每个样本有 n_labels 个标签。y 是一个二进制矩阵，其中每个元素为 0 或 1，表示该样本是否属于某个标签。
# y_pred：预测标签，形状与 y 相同，表示模型对每个样本的预测结果。y_pred 也是二进制矩阵，表示模型预测的每个样本是否属于某个标签。
# scores：预测分数（实数值），形状为 (n_samples, n_labels)，表示模型对每个样本和每个标签的置信度或概率。虽然 scores 在这个函数中没有被使用，但它可能是为了后续可能的扩展或其他用途而传递的参数。
def metric(y, y_pred, scores):
    # 计算 Macro-F1 分数
    # f1_score(y, y_pred, average='macro')：计算 Macro-F1 分数。
    # average='macro'：表示按标签计算每个标签的 F1 分数，然后取平均值。具体来说，Macro-F1 分数是通过以下步骤计算的：
    # 对于每个标签，分别计算其精确率和召回率。
    # 计算每个标签的 F1 分数（即精确率和召回率的调和平均值）。
    # 对所有标签的 F1 分数取平均值。
    # 优点：Macro-F1 分数对每个标签的重要性给予相同的权重，因此它适合处理类别不平衡的情况，尤其是当不同标签的数量差异较大时。
    # 缺点：如果某些标签的样本数量非常少，可能会导致这些标签的 F1 分数对整体结果的影响过大
    macro_f1 = f1_score(y, y_pred, average='macro')
    # 计算 Macro-F1 分数
    # f1_score(y, y_pred, average='micro')：计算 Micro-F1 分数。
    # average='micro'：表示将所有标签的真正例（True Positives, TP）、假正例（False Positives, FP）和假负例（False Negatives, FN）汇总起来，再计算总的精确率和召回率，最后计算 F1 分数。具体来说，Micro-F1 分数是通过以下步骤计算的：
    # 将所有标签的 TP、FP 和 FN 汇总。
    # 计算总的精确率和召回率。
    # 计算总的 F1 分数。
    # 优点：Micro-F1 分数考虑了所有标签的整体表现，特别适合处理多标签分类任务，因为它可以捕捉到模型在所有标签上的总体性能。
    # 缺点：如果某些标签的样本数量远多于其他标签，Micro-F1 分数可能会偏向于那些样本较多的标签。
    micro_f1 = f1_score(y, y_pred, average='micro')
    # 计算 Example-F1 分数
    # f1_score(y, y_pred, average='samples')：计算 Example-F1 分数。
    # average='samples'：表示按样本计算每个样本的 F1 分数，然后取平均值。具体来说，Example-F1 分数是通过以下步骤计算的：
    # 对于每个样本，分别计算其精确率和召回率（基于该样本的所有标签）。
    # 计算每个样本的 F1 分数（即该样本的精确率和召回率的调和平均值）。
    # 对所有样本的 F1 分数取平均值。
    # 优点：Example-F1 分数关注每个样本的标签组合，特别适合处理多标签分类任务，因为它可以评估模型在每个样本上的整体表现。
    # 缺点：如果某些样本的标签数量较少，可能会导致这些样本的 F1 分数对整体结果的影响较小。
    example_f1 = f1_score(y, y_pred, average='samples')

    return micro_f1, macro_f1, example_f1
