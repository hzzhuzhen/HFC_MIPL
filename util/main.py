# numpy：用于数值计算和数组操作。
# scipy.io：用于读取MAT文件（MATLAB格式的数据文件）。
# os：用于操作系统相关的功能，如文件路径操作。
# BinaryRelevance：自定义模块，实现了二元相关性（BR）方法。
# PMLGradientBoosting：自定义模块，实现了部分多标签梯度提升（MLGB）方法。
# metric：自定义模块，用于计算多标签分类的评估指标（Micro-F1、Macro-F1、Example-F1）。
# warnings.filterwarnings("ignore")：忽略所有警告信息，避免输出不必要的警告。
import numpy as np
import scipy.io as sio
import os
from BinaryRelevance import BinaryRelevance
from PMLGradientBoosting import PMLGradientBoosting
from metric import metric
import time
import warnings
from drama_stage_1 import drama_stage_1

warnings.filterwarnings("ignore")
# 实现了一个部分多标签学习（Partial Multi-Label Learning, PML）的实验流程，
# 使用了二元相关性（Binary Relevance, BR）和部分多标签梯度提升（PML Gradient Boosting, MLGB）两种方法来处理数据，并评估它们的性能。

# 2. 加载数据函数 load_data
# 功能：从MAT文件中加载数据，并将其划分为训练集和测试集。训练集占80%（默认），测试集占20%。
# 参数：
#  path：MAT文件的路径。
#  tr_rate：训练集的比例，默认为0.8。
# 返回值：
#  trX：训练集特征。
#  trY：训练集真实标签。
#  trpY：训练集候选标签集（与真实标签相同，但在后续处理中可能会有所不同）。
#  tsX：测试集特征。
#  tsY：测试集真实标签。
def load_data(path, tr_rate=0.8):
    data = sio.loadmat(path) # 取matlab数据
    X = data['X']
    Y = data['Y']
    Y_P = Y # 偏标签数据和Y相同
    data_num = int(X.shape[0]) #样本数
    perm = np.arange(data_num) # 生成序号序列
    np.random.shuffle(perm) # 序号乱序
    X = X[perm] # 用新的序号
    Y = Y[perm] # 用新的序号
    Y_P = Y_P[perm] # 用新的序号
    part = int(data_num * tr_rate) # 取训练数据和测试数据的分割点
    trX = X[0:part, :] # 取训练数据的X
    trY = Y[0:part, :] # 取测试数据的Y
    trpY = Y_P[0:part, :] # 取训练数据的偏标签数据
    tsX = X[part:data_num, :]  # 测试数据的X
    tsY = Y[part:data_num, :]  # 测试数据的Y
    return trX, trY, trY, trpY, tsX, tsY

# 3. 读取预处理数据函数 read_data
#  功能：从预处理的MAT文件中读取训练集和测试集的数据。
#  参数：
#  file_name：MAT文件的名称。
#  返回值：
#  X_train：训练集特征。
#  lmd_train：训练集的置信度矩阵（表示每个标签的置信度）。
#  Y_train：训练集真实标签。
#  Y_P_train：训练集候选标签集。
#  X_test：测试集特征。
#  Y_test：测试集真实标签。
def read_data(file_name):
    data = sio.loadmat(file_name) # 读取matlab数据格式数据
    X_train = data['train_data'] # 读取训练集特征数据
    lmd_train = data['conf'] # 读取重构标签数据文件
    Y_train = data['train_target'] # 读取训练目标标签数据
    Y_P_train = data['train_p_target'] # 读取训练目标偏标签数据
    X_test = data['test_data'] # 读取测试特征数据
    Y_test = data['test_target'] # 读取测试目标标签数据
    return X_train, lmd_train, Y_train, Y_P_train, X_test, Y_test

# 4. 评估函数 evaluate
#  功能：根据预测的分数和真实标签，计算三个评估指标：Micro-F1、Macro-F1 和 Example-F1。
#  参数：
#  Y_test：测试集的真实标签。
#  scores：模型预测的分数（实数值）。
#  返回值：
#  micro_f1：Micro-F1 分数。
#  macro_f1：Macro-F1 分数。
#  example_f1：Example-F1 分数。
def evaluate(Y_test, scores):
    threshold = 0 # 阈值
    micro_f1, macro_f1, example_f1 = metric(
        Y_test, 1 * (scores > threshold), scores) # 得分小于0的，设置为0
    return micro_f1, macro_f1, example_f1 # 输出最小f1值，最大f1值，和样本f1值

# 5. PML梯度提升模型训练和预测函数PML_Gradient_Boosting
# PML_Gradient_Boosting 函数实现了部分多标签学习（Partial Multi-Label Learning, PML）任务中的一种两阶段方法。
# 该方法结合了二元相关性（Binary Relevance, BR）和部分多标签梯度提升（PML Gradient Boosting, MLGB）两个步骤，旨在提高部分多标签数据的预测性能。
# 功能：使用二元相关性（BR）和偏多标签梯度提升（MLGB）模型进行训练和预测。
# 参数：
#  X_train：训练集特征。
#  lmd_train：训练集的置信度矩阵。
#  Y_train：训练集真实标签。
#  Y_P_train：训练集候选标签集。
#  X_test：测试集特征。
#  Y_test：测试集真实标签。
#  target：目标数据集名称（用于日志输出）。
#  返回值：
#  Y_test_pred_0：BR模型对测试集的预测结果。
#  Y_test_pred：PML梯度提升模型对测试集的预测结果。
# 算法的核心思想
# 两阶段策略：
# 第一阶段：使用BR模型对训练集和测试集进行初步预测。BR模型通过为每个标签训练一个独立的SVR模型，得到每个样本的初始标签置信度。这个阶段的主要目的是为后续的梯度提升提供一个合理的起点。
# 第二阶段：使用MLGB模型对初步预测结果进行进一步优化。MLGB模型通过梯度提升的方式，逐步改进预测结果，特别是在处理部分标注数据时，能够有效利用候选标签集中的信息，减少预测误差。
# 梯度提升的优势：
# 逐步改进：梯度提升通过逐步添加弱学习器（如决策树），逐渐减少预测误差。每次迭代中，模型会根据当前的预测误差进行调整，从而不断提高预测精度。
# 处理部分标注数据：MLGB 模型专门针对部分多标签学习设计，能够有效处理候选标签集中的不确定性。通过引入置信度矩阵和候选标签集，MLGB 能够更好地捕捉标签之间的相关性，减少误报和漏报。
# 结合 BR 和 MLGB 的优势：
# BR 提供初步预测：BR 模型为每个标签训练一个独立的模型，能够快速生成初步预测结果。虽然BR忽略了标签之间的相关性，但它提供了一个合理的起点，帮助MLGB模型更快地收敛。
# MLGB 进一步优化：MLGB 模型通过梯度提升的方式，逐步改进预测结果，特别是在处理部分标注数据时，能够有效利用候选标签集中的信息，减少预测误差。MLGB 考虑了标签之间的相关性，能够更好地捕捉复杂的标签依赖关系。
def PML_Gradient_Boosting(X_train, lmd_train, Y_train, Y_P_train, X_test, Y_test, target):
    # 1. 初始化 Binary Relevance (BR) 模型
    # Binary Relevance (BR)：BR是一种经典的多标签分类方法，它将多标签问题分解为多个二分类问题。具体来说，对于每个标签，BR训练一个独立的二分类模型，用于预测该标签是否存在于样本中。
    # base_learner="svr"：在这里，BR使用支持向量回归（Support Vector Regression, SVR）作为基础学习器。
    # SVR 是一种基于支持向量机（SVM）的回归算法，适用于处理非线性可分的数据。通过使用SVR，BR可以更好地捕捉特征与标签之间的复杂关系。
    br = BinaryRelevance(base_learner="svr") # BR算法 二项相关新，其中用的是svr算法
    # 2. 训练 BR 模型
    # X_train：训练集的特征矩阵，形状为 (n_samples, n_features)，表示有 n_samples 个样本，每个样本有 n_features 个特征。
    # lmd_train：训练集的置信度矩阵，形状为 (n_samples, n_labels)，表示每个样本对每个标签的置信度。在部分多标签学习中，lmd_train 包含了每个候选标签的置信度信息，而不是直接的真实标签。这使得 BR 模型能够根据置信度来学习每个标签的分布。
    # 功能：BR 模型根据训练集的特征和置信度矩阵，为每个标签训练一个独立的SVR模型。每个SVR模型的目标是预测该标签的置信度值。
    br.train(X_train, lmd_train)  #【训练一个基础的BR模型】
    # 3. 使用 BR 模型进行初步预测
    # Y_pred_0：BR 模型对训练集的预测结果，形状为 (n_samples, n_labels)，表示每个样本对每个标签的预测置信度。这些预测结果将作为后续 PML 梯度提升模型的输入之一。
    # 功能：BR 模型根据训练好的SVR模型，对训练集进行预测，得到每个样本的初始标签置信度。这些初始预测结果将用于引导后续的梯度提升过程。
    Y_pred_0 = br.predict(X_train)
    # 4. 初始化 PML 梯度提升模型
    # PMLGradientBoosting：这是一个自定义的类，实现了偏多标签梯度提升（PML Gradient Boosting, MLGB）算法。MLGB 是一种专门为偏多标签学习设计的梯度提升方法，能够在处理偏标注数据时有效地捕捉标签之间的相关性。
    # 功能：MLGB 模型通过逐步添加弱学习器（通常是决策树），逐渐改进预测结果。与传统的梯度提升不同，MLGB 针对部分多标签数据的特点，考虑了候选标签集的不确定性，并通过梯度提升的方式逐步减少预测误差。
    mlgb = PMLGradientBoosting()
    # 5. 训练 PML 梯度提升模型
    # X_train：训练集的特征矩阵。
    # lmd_train：训练集的置信度矩阵。
    # Y_P_train：训练集的候选标签集，形状为 (n_samples, n_labels)，表示每个样本的候选标签。
    # 在部分多标签学习中，Y_P_train 包含了每个样本的候选标签集合，但并不是所有候选标签都是真实标签。
    # Y_pred_0：BR 模型对训练集的初步预测结果【BR前面的首次输出】，形状为 (n_samples, n_labels)，表示每个样本对每个标签的初始预测置信度。
    # 功能：MLGB 模型根据训练集的特征、置信度矩阵、候选标签集以及BR模型的初步预测结果，进行训练。MLGB 通过梯度提升的方式，
    # 逐步改进预测结果，特别是在处理部分标注数据时，能够有效利用候选标签集中的信息，减少预测误差。
    mlgb.train(X_train, lmd_train, Y_P_train, Y_pred_0)

    # 6. 使用 BR 模型对测试集进行初步预测
    # X_test：测试集的特征矩阵。
    # Y_test_pred_0：BR 模型对测试集的预测结果，形状为 (n_test_samples, n_labels)，表示每个测试样本对每个标签的预测置信度。
    # 功能：BR 模型根据训练好的SVR模型，对测试集进行预测，得到每个测试样本的初始标签置信度。这些初始预测结果将作为后续 PML 梯度提升模型的输入之一。
    Y_test_pred_0 = br.predict(X_test)
    Y_test_pred = mlgb.predict(X_test, Y_test_pred_0)

    return Y_test_pred_0, Y_test_pred # 输出了br首次预测结果和mlgb的梯度提升预测结果

# 6. 主程序逻辑
# 功能：主程序逻辑，针对指定的目标数据集（如emotion_4），重复运行5次实验，每次实验中使用BR和MLGB模型进行训练和预测，并评估模型的性能。最后，计算并输出每种方法的平均性能和标准差。
#  步骤：
#  设置目标数据集：targets 列表中包含要处理的数据集名称（目前只有一个emotion_4）。
#  循环遍历目标数据集：对于每个目标数据集，读取相应的MAT文件，运行5次实验。
#  每次实验中：
#  调用 read_data 函数读取训练集和测试集数据。
#  调用 PML_Gradient_Boosting 函数进行模型训练和预测。
#  使用 evaluate 函数评估模型的性能，记录BR和MLGB模型的F1分数。
#  打印每次实验的结果。
#  实验结束后：
#  计算BR和MLGB模型的平均性能和标准差。
#  打印最终的平均性能和标准差，分别对应于Naïve-DRAMA（仅使用BR）和DRAMA（使用MLGB）。
if __name__ == '__main__':
    time_s = time.time()
    # python版本的第一阶段manifold
    #target = "baselines/DEMIPL_bak/data/emotion/emotion_4"  # 替换为你的目标文件名
    target = "../data/emotion/emotion_4"  # 替换为你的目标文件名
    drama_stage_1(target)

    # 2. 定义目标数据集
    # targets：这是一个列表，包含要处理的目标数据集名称。当前只有一个数据集 'emotion_4'，但可以扩展为多个数据集。
    # 功能：通过遍历 targets 列表，程序将对每个目标数据集进行实验。
    targets = [
        '../data/emotion/emotion_4'
    ]
    # 3. 循环遍历目标数据集
    # 功能：对于每个目标数据集，程序将执行一系列实验操作。target 是当前处理的数据集名称。
    for target in targets:
        # 4. 构建文件名并初始化评估指标列表
        # file_name：根据目标数据集名称 target，构建对应的MAT文件名。例如，'emotion_4' 对应的文件名为 'emotion_4_conf.mat'。
        # micro、macro、example：用于存储每次实验中使用 MLGB 模型 的 Micro-F1、Macro-F1 和 Example-F1 评估结果。
        # micro_0、macro_0、example_0：用于存储每次实验中使用 BR 模型 的 Micro-F1、Macro-F1 和 Example-F1 评估结果。
        # 功能：这些列表将用于记录每次实验的评估结果，后续会计算平均值和标准差。
        file_name = target + "_conf_py.mat"
        micro = []
        macro = []
        example = []
        micro_0 = []
        macro_0 = []
        example_0 = []
        # 5. 多次实验循环
        # 功能：对于每个目标数据集，程序将重复运行 5次 实验。这样做是为了确保实验结果的稳定性和可靠性，避免由于数据划分或随机性带来的偶然误差。
        for i in range(5):
            # 6. 读取数据
            # read_data(file_name)：调用 read_data 函数从指定的MAT文件中读取训练集和测试集的数据。
            # 返回值：
            # X_train：训练集的特征矩阵。
            # lmd_train：训练集的置信度矩阵（表示每个候选标签的置信度）。
            # Y_train：训练集的真实标签。
            # Y_P_train：训练集的候选标签集。
            # X_test：测试集的特征矩阵。
            # Y_test：测试集的真实标签。
            # 功能：每次实验中，都会重新读取数据，确保每次实验使用的是独立的数据划分。
            X_train, lmd_train, Y_train, Y_P_train, X_test, Y_test = read_data(file_name)
            # lmd_train = Y_train  # 暂时用重构前的标签【用于测试原标签效果】
            # 7. 调用 PML 梯度提升模型进行预测
            # PML_Gradient_Boosting：调用之前解释过的 PML_Gradient_Boosting 函数，传入训练集和测试集的数据，返回两个预测结果：
            # Y_pred_0：BR 模型对测试集的初步预测结果。
            # Y_pred：MLGB 模型对测试集的最终预测结果。
            # 功能：通过 BR 和 MLGB 两种模型分别对测试集进行预测，得到两个不同的预测结果。
            Y_pred_0, Y_pred = PML_Gradient_Boosting(X_train, lmd_train, Y_train, Y_P_train, X_test, Y_test, target)
            # 8. 评估模型性能
            # evaluate(Y_test, Y_pred)：调用 evaluate 函数，传入真实标签 Y_test 和预测结果 Y_pred，计算三个评估指标：Micro-F1、Macro-F1 和 Example-F1。
            # evaluate(Y_test, Y_pred_0)：同样调用 evaluate 函数，但传入的是 BR 模型的初步预测结果 Y_pred_0，计算 BR 模型的评估指标。
            # 功能：分别评估 MLGB 模型和 BR 模型的性能，并将结果存储在 micro_f1、macro_f1、example_f1 和 micro_f1_0、macro_f1_0、example_f1_0 中。
            micro_f1, macro_f1, example_f1 = evaluate(Y_test, Y_pred)
            micro_f1_0, macro_f1_0, example_f1_0 = evaluate(Y_test, Y_pred_0)
            # 9. 记录评估结果
            # 功能：将每次实验的评估结果（MLGB 和 BR 模型的 F1 分数）分别添加到对应的列表中，以便后续计算平均值和标准差。
            micro.append(micro_f1)
            macro.append(macro_f1)
            example.append(example_f1)
            micro_0.append(micro_f1_0)
            macro_0.append(macro_f1_0)
            example_0.append(example_f1_0)
            # 10. 打印每次实验的结果
            # 功能：每次实验结束后，打印当前实验的编号（i）以及 MLGB 模型的评估结果（Micro-F1、Macro-F1 和 Example-F1）。
            print('%s\t%.4f\t%.4f\t%.4f' % (target+'iter: '+str(i), micro_f1, macro_f1, example_f1))
            # print('%s\t%.4f\t%.4f\t%.4f' % (target+'iter: '+str(i), micro_f1_0, macro_f1_0, example_f1_0))
        # 11. 计算平均值和标准差
        # 功能：将所有实验的评估结果转换为 NumPy 数组，并计算每个评估指标的 平均值 和 标准差。ddof=1 表示计算样本标准差时使用无偏估计（即除以 n-1 而不是 n）。
        # macro_mean、micro_mean、example_mean：分别是 MLGB 模型的 Macro-F1、Micro-F1 和 Example-F1 的平均值。
        # macro_std、micro_std、example_std：分别是 MLGB 模型的 Macro-F1、Micro-F1 和 Example-F1 的标准差。
        # macro_mean_0、micro_mean_0、example_mean_0：分别是 BR 模型的 Macro-F1、Micro-F1 和 Example-F1 的平均值。
        # macro_std_0、micro_std_0、example_std_0：分别是 BR 模型的 Macro-F1、Micro-F1 和 Example-F1 的标准差。
        macro = np.array(macro)
        micro = np.array(micro)
        example = np.array(example)
        macro_0 = np.array(macro_0)
        micro_0 = np.array(micro_0)
        example_0 = np.array(example_0)
        macro_std = np.std(macro, ddof=1)
        micro_std = np.std(micro, ddof=1)
        example_std = np.std(example, ddof=1)
        macro_mean = np.mean(macro)
        micro_mean = np.mean(micro)
        example_mean = np.mean(example)
        macro_std_0 = np.std(macro_0, ddof=1)
        micro_std_0 = np.std(micro_0, ddof=1)
        example_std_0 = np.std(example_0, ddof=1)
        macro_mean_0 = np.mean(macro_0)
        micro_mean_0 = np.mean(micro_0)
        example_mean_0 = np.mean(example_0)

        # 12. 打印最终结果
        # 功能：打印最终的实验结果，包括：
        # Naïve-DRAMA（仅使用 BR 模型）：输出 BR 模型的平均性能和标准差。
        # DRAMA（使用 MLGB 模型）：输出 MLGB 模型的平均性能和标准差。
        # 格式：每行输出包含两部分，分别是平均值和标准差，便于比较两种模型的性能差异。
        print('%s\t%.4f\t%.4f\t%.4f\t \t%s\t%.4f\t%.4f\t%.4f' % (
            'ndrama: ' + target + ' / mean', micro_mean_0, macro_mean_0, example_mean_0, target + ' / std', micro_std_0,
            macro_std_0, example_std_0))
        print('%s\t%.4f\t%.4f\t%.4f\t \t%s\t%.4f\t%.4f\t%.4f' % (
            'drama: ' + target + ' / mean', micro_mean, macro_mean, example_mean, target + ' / std', micro_std, macro_std,
            example_std))
        # print('%s\t%.4f\t%.4f\t%.4f' % (target, micro_mean, macro_mean, example_mean))
