# sklearn.svm.SVR：支持向量回归（Support Vector Regression, SVR），适用于非线性回归任务。
# sklearn.tree.DecisionTreeRegressor：决策树回归器（Decision Tree Regressor），适用于回归任务，能够捕捉特征与标签之间的复杂关系。
# sklearn.kernel_ridge.KernelRidge：核岭回归（Kernel Ridge Regression, KRR），结合了岭回归和核方法，适用于非线性回归任务。
# numpy：用于数值计算和数组操作，特别是在处理矩阵和向量时非常有用。
from sklearn.svm import SVR
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
import numpy as np
###################
# 定义 BinaryRelevance 类
# 这段代码实现了一个名为 BinaryRelevance 的类，用于处理多标签分类任务中的二元相关性（Binary Relevance, BR）方法。
# BR 是一种经典的多标签学习策略，它将多标签问题分解为多个独立的二分类或回归问题，每个标签对应一个单独的模型。
####################
class BinaryRelevance:
    # __init__ 方法：
    # 参数：
    # base_learner：指定基础学习器的类型，默认值为 "cart"，表示使用决策树回归器（CART）。
    # 其他可选的基础学习器包括 "kernel_lr"（核岭回归）和 "svr"（支持向量回归）。
    # 功能：初始化 BinaryRelevance 类，并设置基础学习器的类型。self.base_learner 保存了用户指定的基础学习器类型。
    def __init__(self, base_learner="cart"):
        self.base_learner = base_learner

    # 3. 训练方法 train
    # train 方法：
    # 参数：
    # X：训练集的特征矩阵，形状为 (n_samples, n_features)，表示有 n_samples 个样本，每个样本有 n_features 个特征。
    # Y：训练集的标签矩阵，形状为 (n_samples, n_labels)，表示每个样本对 n_labels 个标签的置信度或真实值。
    def train(self, X, Y):
        # 确定标签数量：self.n_labels = Y.shape[1] 获取标签的数量 n_labels，即 Y 的列数。
        self.n_labels = Y.shape[1]
        # 初始化回归器列表：self.regressors = [] 创建一个空列表，用于存储每个标签对应的回归模型。
        self.regressors = []
        # 选择基础学习器：根据 self.base_learner 的值，为每个标签选择合适的基础学习器：
        for i in range(self.n_labels):
            # 如果 base_learner 是 "kernel_lr"，则使用 KernelRidge 核岭回归器。
            if self.base_learner == "kernel_lr":
                self.regressors.append(KernelRidge(alpha=1.0))
            # 如果 base_learner 是 "cart"，则使用 DecisionTreeRegressor 决策树回归器。
            elif self.base_learner == "cart":
                self.regressors.append(tree.DecisionTreeRegressor())
            # 否则，默认使用 SVR 支持向量回归器，带有 RBF 核函数，C=10.0 和 epsilon=0.1
            else:
                self.regressors.append(SVR(kernel="rbf", C=10.0, epsilon=0.1))
                # self.regressors.append(SVR(C=100.0, epsilon=0.001))

        # 训练每个回归器：对于每个标签 i，使用 self.regressors[i].fit(X, Y[:, i]) 训练相应的回归模型。
        # Y[:, i] 表示第 i 个标签的置信度或真实值，作为该标签的训练目标。
        for i in range(self.n_labels):
            self.regressors[i].fit(X, Y[:, i])

    # 4. 预测方法 predict
    # predict 方法：
    # 参数：
    # X_test：测试集的特征矩阵，形状为 (n_test_samples, n_features)，表示有 n_test_samples 个测试样本，每个样本有 n_features 个特征。
    def predict(self, X_test):
        # 初始化预测矩阵：Y_pred = np.zeros((X_test.shape[0], self.n_labels)) 创建一个形状为 (n_test_samples, n_labels) 的零矩阵，用于存储每个测试样本对每个标签的预测结果。
        Y_pred = np.zeros((X_test.shape[0], self.n_labels))
        # 进行预测：对于每个标签 i，使用 self.regressors[i].predict(X_test) 对测试集进行预测，
        # 并将结果存储在 Y_pred[:, i] 中。self.regressors[i] 是之前训练好的回归模型，负责预测第 i 个标签的置信度。
        for i in range(self.n_labels):
            Y_pred[:, i] = self.regressors[i].predict(X_test)
        # 返回预测结果：最终返回 Y_pred，这是一个形状为 (n_test_samples, n_labels) 的矩阵，表示每个测试样本对每个标签的预测置信度。
        return Y_pred
