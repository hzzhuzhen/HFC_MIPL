####################
#  PMLGradientBoosting 是一个自定义的类，实现了部分多标签学习（Partial Multi-Label Learning, PML）任务中的梯度提升（Gradient Boosting）算法。
#  该类通过逐步添加弱学习器（如决策树），逐渐改进预测结果，特别是在处理部分标注数据时，能够有效利用候选标签集中的信息，减少预测误差。
####################
# numpy：用于数值计算和数组操作，特别是矩阵运算。
# sklearn.tree.DecisionTreeRegressor：这是 Scikit-learn 库中的一个类，
# 用于构建决策树回归模型。在梯度提升中，通常使用决策树作为弱学习器。
import numpy as np
from sklearn import tree

# PMLGradientBoosting 是一个类，封装了偏多标签梯度提升算法的训练和预测过程。
# 该类允许用户指定基础学习器（base_learner）和最大迭代次数（max_iter），并提供了 train 和 predict 两个主要方法。
class PMLGradientBoosting:
    # 3. 初始化方法 __init__
    # 参数：
    # base_learner：基础学习器的类型，默认为 "svr"（支持向量回归）。虽然在这个实现中没有直接使用 base_learner 参数，但它可能是为了后续扩展或其他用途而设计的。
    # max_iter：最大迭代次数，默认为 10。每次迭代中，模型会添加一个新的弱学习器（如决策树），以逐步改进预测结果。
    # 功能：初始化类的属性，包括基础学习器类型和最大迭代次数。这些参数将在训练过程中使用。
    def __init__(self, base_learner="svr", max_iter=10):
        self.base_learner = base_learner
        self.max_iter = max_iter

    # 4. 训练方法 train
    # 参数：
    # X：训练集的特征矩阵，形状为 (n_samples, n_features)，表示有 n_samples 个样本，每个样本有 n_features 个特征。
    # lmd：训练集的置信度矩阵，形状为 (n_samples, n_labels)，表示每个样本对每个标签的置信度。在偏多标签学习中，lmd 包含了每个候选标签的置信度信息。
    # Y_P【没实际用起来】：训练集的候选标签集，形状为 (n_samples, n_labels)，表示每个样本的候选标签。在偏多标签学习中，Y_P 包含了每个样本的候选标签集合【训练标签的候选标签集】，但并不是所有候选标签都是真实标签。
    # Y_pred_0：BR 模型对训练集的初步预测结果，形状为 (n_samples, n_labels)，表示每个样本对每个标签的初始预测置信度。
    # 功能：训练 PML 梯度提升模型，逐步添加弱学习器（如决策树），以逐步改进预测结果。
    def train(self, X, lmd, Y_P, Y_pred_0):
        # 4.1 初始化变量
        # self.regressors：用于存储每次迭代中训练的弱学习器（如决策树）。每个弱学习器都用于预测残差，并在下一次迭代中更新预测结果。
        self.regressors = []
        # self.alphas：用于存储每次迭代中计算的学习率（alpha）。学习率控制着每次迭代中弱学习器的贡献，防止过拟合。
        self.alphas = []
        # Y_pred_prev：【前面预测器叠加预测结果】初始化为 BR 模型的初步预测结果 Y_pred_0，表示当前的预测值。每次迭代后，Y_pred_prev 都会根据新添加的弱学习器进行更新。
        Y_pred_prev = Y_pred_0

        # 4.2 处理候选标签集
        # Y_P：将置信度矩阵 lmd 转换为二进制矩阵，表示每个样本的候选标签。具体来说，如果某个标签的置信度大于 0，则将其标记为 1，否则标记为 0。这样可以将候选标签集转换为二进制形式，便于后续计算。
        Y_P = 1.0 * (lmd > 0)  # 通过置信度赋值给Y_P 代表候选标签
        # Y_P_neg：计算候选标签集的补集，即 1 - Y_P。这表示每个样本不属于候选标签的概率。
        Y_P_neg = 1 - Y_P #不属于候选标签集的概率

        # 4.3 梯度提升迭代
        for i in range(self.max_iter):
            # neg_gradient：计算负梯度（负残差），即 lmd - Y_pred_prev。负梯度表示当前预测值与真实置信度之间的差异，反映了模型需要改进的方向。
            neg_gradient = lmd - Y_pred_prev # 模型拟合的方向
            # if np.sum(neg_gradient) == 0：如果负梯度的总和为 0，说明模型已经完全拟合了数据，此时可以提前终止迭代，避免不必要的计算。
            if np.sum(neg_gradient) == 0:
                break
            # 4.4 训练弱学习器
            # Z：将特征矩阵 X 和当前预测值 Y_pred_prev 拼接在一起，形成新的输入特征矩阵 Z。这样做是为了让弱学习器不仅考虑原始特征，还考虑当前的预测值，从而更好地捕捉标签之间的相关性。
            Z = np.hstack((X, Y_pred_prev)) #居然把特征X和前几次迭代的预测结果Y_pred_prev，全部拼接乘输入特征。
            # regressor：创建一个决策树回归器，最大深度为 5。决策树是一种常用的弱学习器，能够在每次迭代中拟合负梯度，帮助模型逐步改进预测结果。
            regressor = tree.DecisionTreeRegressor(max_depth=5) # 5层深的决策数
            # regressor.fit(Z, neg_gradient)：使用负梯度作为目标变量，训练决策树回归器。这样，弱学习器将学会如何修正当前预测值中的误差。
            regressor.fit(Z, neg_gradient) # 参数拟合负梯度的方向的残差。
            # fit data from both X and Y

            # 4.5 计算学习率 alpha
            # R：弱学习器对输入特征 Z 的预测结果，表示它对负梯度的拟合程度。
            R = regressor.predict(Z) #用R来拟合负梯度方向的残差
            # calculate the predicted residual value
            # Delta：负梯度 neg_gradient，表示当前预测值与真实置信度之间的差异。
            Delta = neg_gradient
            # a 和 b：通过计算 R 和 Delta 的内积，得到两个标量 a 和 b。这两个标量用于计算学习率 alpha，以控制弱学习器的贡献。
            a = np.trace(R.T.dot(R))
            b = - 2 * np.trace(R.T.dot(Delta))
            # c = np.trace(Delta.T.dot(Delta))
            # alpha：估计是论文中公式（4）的求导结构，学习率的计算公式为 - b / (2 * a)。学习率的作用是防止模型过拟合，确保每次迭代中弱学习器的贡献适中。
            alpha = - b / (2 * a)
            # calculate alpha to prevent overfitting
            # 4.6 更新预测值
            # 功能：根据弱学习器的预测结果 R 和学习率 alpha，更新当前的预测值 Y_pred_prev。这样，模型在每次迭代后都会逐步改进预测结果，逐渐接近真实置信度
            Y_pred_prev = Y_pred_prev + alpha * R
            # update the prediction

            # 4.7 保存弱学习器和学习率
            # 功能：将当前迭代中训练的弱学习器 regressor
            # 和计算的学习率 alpha 保存到类的属性中，以便在预测阶段使用。
            self.regressors.append(regressor)
            self.alphas.append(alpha)
            # save the regressor for testing

        # 4.8 记录学习器数量
        # 功能：记录最终训练的弱学习器数量，即 self.learner_count。这个值将在预测阶段用于控制迭代次数。
        self.learner_count = len(self.regressors)

    # 5. 预测方法 predict
    # 参数：
    # X_test：测试集的特征矩阵，形状为 (n_test_samples, n_features)，表示有 n_test_samples 个测试样本，每个样本有 n_features 个特征。
    # Y_pred_0：BR 模型对测试集的初步预测结果，形状为 (n_test_samples, n_labels)，表示每个测试样本对每个标签的初始预测置信度。
    # 功能：使用训练好的 PML 梯度提升模型对测试集进行预测，逐步添加弱学习器的贡献，最终输出改进后的预测结果。
    def predict(self, X_test, Y_pred_0):
        # 5.1 初始化预测值
        # 功能：将 BR 模型的初步预测结果 Y_pred_0 复制到 Y_pred 中，作为初始预测值。Y_pred 将在后续迭代中逐步更新。
        Y_pred = np.copy(Y_pred_0)
        # 5.2 逐个应用弱学习器

        for i in range(self.learner_count):
            # Z_hat：将测试集的特征矩阵 X_test 和当前预测值 Y_pred 拼接在一起，形成新的输入特征矩阵 Z_hat。
            # 这样做是为了让弱学习器不仅考虑原始特征，还考虑当前的预测值，从而更好地捕捉标签之间的相关性。
            Z_hat = np.hstack((X_test, Y_pred))
            # self.regressors[i].predict(Z_hat)：使用第 i 个弱学习器对输入特征 Z_hat 进行预测，得到该弱学习器对负梯度的拟合结果。
            # Y_pred += self.alphas[i] * ...：根据弱学习器的预测结果和对应的学习率 alpha，更新当前
            Y_pred += self.alphas[i] * self.regressors[i].predict(Z_hat)
        return Y_pred


# 补充
# 学习率计算公式及其意义
# 你提到的学习率计算公式是梯度提升（Gradient Boosting）算法中的一部分，特别是在部分多标签学习（Partial Multi-Label Learning, PML）任务中用于控制弱学习器的贡献。这段代码的核心思想是通过最小化损失函数来更新模型的预测值，并使用学习率 alpha 来调节每次迭代中弱学习器的贡献，防止过拟合。
#
# 1. 背景：梯度提升的基本原理
# 梯度提升是一种集成学习方法，它通过逐步添加弱学习器（如决策树）来改进模型的预测性能。每一轮迭代中，模型会根据当前预测与真实值之间的残差（即负梯度 neg_gradient），训练一个新的弱学习器来拟合这些残差。然后，将该弱学习器的预测结果以一定的权重（学习率 alpha）加到当前的预测值上，从而逐步减小误差。
#
# 2. 关键变量解释
# Delta = neg_gradient：
# neg_gradient 是负梯度，表示当前预测值与真实值之间的残差。在梯度提升中，负梯度可以看作是模型需要改进的方向。Delta 保存了这个负梯度，表示模型需要调整的部分。
# R：
# R 是弱学习器的预测结果。在每一轮迭代中，弱学习器会根据当前的残差训练一个模型，并输出预测值 R。R 的形状与 Y_pred_prev 相同，表示每个样本对每个标签的预测置信度。
# Y_pred_prev：
# Y_pred_prev 是当前的预测值，表示模型在前一轮迭代中的预测结果。每次迭代后，Y_pred_prev 都会被更新，逐渐接近真实置信度。
# 3. 学习率 alpha 的计算公式
# python
# 深色版本
# a = np.trace(R.T.dot(R))
# b = - 2 * np.trace(R.T.dot(Delta))
# alpha = - b / (2 * a)
# 3.1 计算 a 和 b
# a = np.trace(R.T.dot(R))：
# R.T.dot(R) 计算的是弱学习器预测结果 R 的自内积矩阵。np.trace 函数计算该矩阵的迹（即主对角线元素之和）。a 实际上是 R 的平方和，反映了弱学习器预测结果的方差大小。
# 意义：a 表示弱学习器预测结果的强度。如果 a 较大，说明弱学习器的预测结果较为强烈，可能需要较小的学习率来避免过度修正；如果 a 较小，说明弱学习器的预测结果较弱，可以使用较大的学习率来加速收敛。
# b = - 2 * np.trace(R.T.dot(Delta))：
# R.T.dot(Delta) 计算的是弱学习器预测结果 R 与负梯度 Delta 之间的内积矩阵。np.trace 函数计算该矩阵的迹，得到一个标量 b。
# 意义：b 表示弱学习器预测结果与负梯度之间的相关性。如果 b 较大，说明弱学习器的预测方向与负梯度方向一致，能够有效减少误差；如果 b 较小或为负，说明弱学习器的预测方向与负梯度方向不一致，可能会增加误差。
# 3.2 学习率 alpha 的计算
# alpha = - b / (2 * a)：
# 这个公式来自于二次函数的极值点公式。假设我们有一个二次损失函数 L(Y_pred) = a * alpha^2 + b * alpha + c，其中 a 和 b 是通过内积计算得到的标量，c 是常数项（在这个例子中没有显式计算，因为它不影响 alpha 的最优值）。
# 为了最小化损失函数 L(Y_pred)，我们需要找到使 dL/dalpha = 0 的 alpha 值。通过对 L(Y_pred) 求导并令其等于零，可以得到最优的学习率 alpha = - b / (2 * a)。
# 意义：alpha 是一个动态调整的学习率，它根据弱学习器的预测结果 R 和负梯度 Delta 之间的关系自动调整。当弱学习器的预测方向与负梯度方向一致时，alpha 会较大，允许模型更快地收敛；当弱学习器的预测方向与负梯度方向不一致时，alpha 会较小，防止模型过度修正。
# 4. 防止过拟合
# python
# 深色版本
# # calculate alpha to prevent overfitting
# 防止过拟合：学习率 alpha 的作用之一是防止模型过拟合。过大的学习率可能导致模型在某一轮迭代中对弱学习器的预测结果进行过度修正，从而使模型过于依赖当前的弱学习器，导致过拟合。通过动态调整 alpha，可以确保每次迭代中弱学习器的贡献适中，既能够有效减少误差，又不会过度修正，从而提高模型的泛化能力。
# 5. 更新预测值
# python
# 深色版本
# Y_pred_prev = Y_pred_prev + alpha * R
# 功能：根据弱学习器的预测结果 R 和学习率 alpha，更新当前的预测值 Y_pred_prev。这样，模型在每次迭代后都会逐步改进预测结果，逐渐接近真实置信度。
# 意义：通过逐步更新预测值，模型可以在多次迭代中逐渐逼近最优解。每次迭代中，弱学习器的预测结果都会被加权（乘以 alpha）并累加到当前的预测值上，从而逐步减小误差。
# 6. 总结
# 学习率 alpha 的计算公式：alpha = - b / (2 * a) 是基于二次损失函数的极值点公式推导出来的。它通过计算弱学习器预测结果 R 和负梯度 Delta 之间的内积，动态调整学习率，确保每次迭代中弱学习器的贡献适中，防止过拟合。
# a 和 b 的意义：a 反映了弱学习器预测结果的强度，b 反映了弱学习器预测结果与负梯度之间的相关性。通过这两个标量，可以有效地控制学习率的大小，确保模型在每次迭代中都能朝着正确的方向前进。
# 防止过拟合：学习率 alpha 的动态调整有助于防止模型过拟合，确保模型在多次迭代中逐步逼近最优解，而不是在某一轮迭代中过度修正。