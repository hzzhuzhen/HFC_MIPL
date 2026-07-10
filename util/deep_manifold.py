import numpy as np
import torch
from faiss import normalize_L2
from sklearn import preprocessing
import scipy
import scipy.stats
import time
import faiss
from sklearn.neighbors import kneighbors_graph
import os



# 3. 估计标签相关矩阵[计算不同标签再样本中共线出现在总它的总共线中的比例，计算该标签相关性。这个矩阵可以用来衡量不同标签之间的关联程度]
# 功能：该函数用于估计标签之间的相关性矩阵L。输入 Y_P 是一个形状为 (n, num_class) 的矩阵，表示每个样本的标签分布。
# 步骤：
# （1）.初始化一个全零矩阵R，用于存储标签之间的相关性。
# （2）.对于每一对标签i和j，如果它们不是同一个标签（即 i!=j），计算它们之间的相关性。【相关性是通过两个标签向量的点积除以它们的总和】来计算的。
# （3）.如果某个标签的总和为零，为了避免除零错误，将其相关性设为 1e-5。
# （4）.最后，通过归一化操作将 R 转换为对称正定矩阵 L，并返回 L。
def estimating_label_correlation_matrix(Y_P):
	num_class = Y_P.shape[1]  # 样本类别
	n = Y_P.shape[0]  # 样本数

	R = np.zeros((num_class, num_class)) #R矩阵 L*L，其中L是标签的数量。这个矩阵将用于存储标签之间的相关性值。
	for i in range(num_class):
		for j in range(num_class):
			# 对角线元素：当 i == j 时，R[i][j] 被设置为 0。这表示同一个标签之间的相关性为 0（即不考虑标签自身的相关性）
			if i == j:
				R[i][j] = 0
			# 非对角线元素：当i!=j时，计算标签i和标签j之间的相关性。具体计算公式为：
			else:
				if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0 :
					R[i][j] = 1e-5 # 设置一个很小的数，防止除不尽。avoid divide zero error
				else:
					# Y_P[:, i] 表示所有样本中标签i的标注情况（0 或 1），Y_P[:, j] 表示所有样本中标签j 的标注情况（0 或 1）。
					# Y_P[:, i].dot(Y_P[:, j]) 计算的是标签i和标签j同时出现的样本数。
					# Y_P[:, i].sum() 和 Y_P[:, j].sum() 分别是标签i和标签j 出现的总次数。通过除以两个标签出现次数之和，归一化了它们的共现频率。
					R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum()) # 对应到论文公式中求a^y_ij并求A_y
	row_sums = np.sum(R, axis=1)
	row_sums[row_sums == 0] = 1e-5  # 替换零值
	D_1_2 = np.diag(1. / np.sqrt(row_sums))  # 计算的是每一行的和，表示每个标签与其他所有标签的相关性之和，将这些值放在对角线上，构建一个对角矩阵。
	L = D_1_2.dot(R).dot(D_1_2) # D_1_2对R点乘，然后再对D_1_2点乘。
	# np.nan_to_num 函数函数用于将数组中的 NaN 和无穷大值替换为合理的数值。
	# 具体来说： NaN 被替换为 0。正无穷大被替换为最大有限浮点数。负无穷大被替换为最小有限浮点数。
	L = np.nan_to_num(L)
	# 返回归一化的标签相关性矩阵L，它是一个L×L 的矩阵，表示标签之间的相关性。
	return L


# 5. 构建图（版本1，使用 Faiss）【K近邻构建领接矩阵构图，向量距离越近度越少权重越高，从而计算X相似】
# 功能：该函数使用 Faiss 库进行高效的 kNN 搜索，构建样本之间的相似性图。
# 用于构建一个基于 k-近邻（kNN）搜索的图，并对图的邻接矩阵进行归一化处理。这个图可以用于图神经网络（GNN）、谱聚类等任务中。
# 参数 ：
# X：形状为 (N, d) 的特征矩阵，其中 N 是样本数量，d 是每个样本的特征维度。
# k：每个节点的 k 近邻数，默认值为 10。
# args：包含一些配置参数的对象，例如是否打印日志信息（no_verbose）。
# 步骤：
# 将输入数据 X 转换为 float32 类型，并构建 Faiss 的 GPU 索引。
# 对数据进行 L2 归一化，并将数据添加到索引中。
# 使用 Faiss 进行 kNN 搜索，获取每个样本的最近邻及其距离。
# 根据搜索结果构建稀疏相似性图 W，并对图进行归一化处理。
def build_graph(X, k=3,args=None):
	if not args.no_verbose:
		print('Building Graph - V2 is not used')
	# (a)KNN图检索,其中把特征传进来，k是近邻数量 kNN search for the graph
	W = kneighbors_graph(X, k, mode='distance', include_self=False, metric='euclidean')

	# (b)应用高斯核函数（Gaussian Kernel Function），其数学表达式为：exp(-距离^2/(2*标准差的平方))
	# 其中 W 是一个稀疏矩阵，通常使用 scipy.sparse 库中的稀疏矩阵格式（如 csr_matrix）来表示图的邻接矩阵，
	# 其中 W.data 是一个一维数组，包含了稀疏矩阵中所有非零元素的值。这些值在初始情况下通常是 k 近邻搜索得到的距离值。
	# 其中 W.data ** 2：这一步将 W.data 中的所有距离值平方。平方操作是为了放大距离的差异，使得距离较近的点之间的相似度更高，而距离较远的点之间的相似度更低
	W.data = np.exp(- W.data ** 2 / (2 * 1))

	# (c)对称化：确保图的邻接矩阵是对称的，适用于无向图。这一步将W和W.T相加，确保邻接矩阵是对称的。因为在无向图中，如果节点i和节点j之间有边，则𝑊_ij=W_ji。
	# 通过加上转置矩阵，可以保证即使原始的W不是对称的，最终的W也是对称的。
	W = W + W.T
	# (d)去除自环：确保邻接矩阵中没有自环（即节点与自身之间的边）。在图传播中，通常不希望节点与自身相连，因此需要移除对角线上的非零值。
	# 将 W 减去这个对角矩阵，实际上相当于将 W 的对角线元素全部设为 0。
	W = W - scipy.sparse.diags(W.diagonal())
	# (e)用度构建对角矩阵D
	# 数组S实际上是每个节点的度（Degree），即每个节点连接的邻居数量。W.sum(axis=1) 沿着每一行求和，返回一个包含每行元素之和的一维数组。
	S = W.sum(axis=1)
	# 去除0，如果某个节点的度为 0（即该节点没有邻居），则将其度设置为 1:这一步是为了防止后续计算中出现除以零的情况。
	S[S == 0] = 1
	# 计算度矩阵的平方根倒数：用于后续的归一化操作。
	D = np.array(1. / np.sqrt(S))
	# 构建度矩阵的平方根倒数矩阵：D 是一个对角矩阵，用于后续的归一化操作。其中 D.reshape(-1)将矩阵转成数组
	D = scipy.sparse.diags(D.reshape(-1))
	# (f)这一步是将邻接矩阵 W 与度矩阵 D 进行两次相乘，得到归一化的邻接矩阵Wn
	# 这种归一化方式称为对称归一化，它使得每个节点与其邻居之间的相似度权重被节点的度所调整，从而更好地反映节点之间的相对重要性。
	Wn = D * W * D

	return Wn

# 6. 标签传播【标签的更新】
# 功能：该函数实现了标签传播算法，通过图传播更新标签分布。
# 标签传播算法（Label Propagation Algorithm, LPA），用于通过图结构传播标签信息，并更新标签分布。该算法结合了图的邻接矩阵、初始标签分布、模型预测以及标签相关性矩阵，通过迭代优化标签分布。
# 参数说明：
# 输入：
# args：包含一些配置参数的对象，例如学习率 gamma。
# Wn：归一化的邻接矩阵，表示图中节点之间的相似度或连接强度。
# L：标签相关性矩阵，表示不同标签之间的关联性。
# Y_pred：模型预测的标签分布。
# Y_P_train：训练集中的初始标签分布（部分标记数据）。
# Z_current：当前的标签分布，通常初始化为 Y_P_train。
# alpha：控制基于图结构传播项的权重。
# eta：控制基于初始标签分布约束项的权重。
# beta：控制基于标签相关性约束项的权重。
# maxiter：最大迭代次数。
# 步骤：
# 将输入的标签分布 Z_current 和其他相关矩阵转换为 PyTorch 张量，并移动到 GPU 上。
# 在 maxiter 次迭代中，计算梯度并更新标签分布 Z_g。梯度由四个部分组成：
# alpha * (Z_g - W_matmul_Z_g)：基于图结构的传播项。
# eta * (Z_g - Y_P_train_g)：基于初始标签分布的约束项。
# 1 * (Z_g - Y_pred_g)：基于模型预测的约束项。
# beta * (Z_g - Z_g @ L_g)：基于标签相关性的约束项。
# 更新后的标签分布 Z 经过 Min-Max 归一化后返回。
def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter):

	# 2. 初始化标签分布
	# gamma 是学习率，控制每次更新标签分布时的步长。
	gamma = args.lr  # learning rate
	# Z 初始化为 Y_P_train，即训练集中的初始标签分布。
	Z = Y_P_train

	# 3. 将输入数据转换为 PyTorch 张量并移动到 GPU
	# 将 Z、Y_P_train、Y_pred 和 L 转换为 PyTorch 的张量，并将它们移动到 GPU 上进行加速计算。
	# torch.from_numpy(...) 将 NumPy 数组转换为 PyTorch 张量。 .float() 将张量的数据类型转换为 float32，以节省内存并提高计算效率。.detach() 断开张量与计算图的连接，避免梯度追踪。.cuda() 将张量移动到 GPU 上。
	Z_g = torch.from_numpy(Z).float().detach().cuda()
	Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
	Y_pred_g =torch.from_numpy(Y_pred).float().detach().cuda()
	L_g = torch.from_numpy(L).float().detach().cuda()

	# 4. 标签传播迭代
	with torch.no_grad():
		for i in range(maxiter):
			# W_matmul_Z_g：计算图传播项 W⋅Z，即将当前标签分布 Z_g 通过图的邻接矩阵 Wn 进行传播。这里使用 Wn.dot(Z_g.cpu().numpy()) 将 Z_g 移回 CPU 进行矩阵乘法，然后再移回 GPU。
			# grad：计算梯度，梯度由四个部分组成：
			# 基于图结构的传播项：alpha * (Z_g - W_matmul_Z_g)，鼓励标签在图上平滑传播，使得相邻节点的标签分布更加相似。
			# 基于初始标签分布的约束项：eta * (Z_g - Y_P_train_g)，确保标签分布不会偏离初始标签分布太远。
			# 基于模型预测的约束项：1 * (Z_g - Y_pred_g)，确保标签分布接近模型的预测结果。
			# 基于标签相关性的约束项：beta * (Z_g - Z_g @ L_g)，利用标签相关性矩阵 L 来调整标签分布，使得相关标签之间的分布更加一致。
			# 更新标签分布：Z_g = Z_g - gamma * grad，根据计算的梯度更新标签分布 Z_g，步长由学习率 gamma 控制。
			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).float().detach().cuda()
			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
			Z_g = Z_g - gamma * grad

	# 5. 将更新后的标签分布移回 CPU 并转换为 NumPy 数组
	Z = Z_g.detach().cpu().numpy()
	# 6. Min-Max 归一化
	# 使用 MinMaxScaler 对标签分布 Z 进行 Min-Max 归一化，将每个标签的值缩放到 [0, 1] 区间内。这一步有助于确保标签分布的数值范围合理，避免极端值对后续任务的影响。
	min_max_scaler = preprocessing.MinMaxScaler()
	Z = min_max_scaler.fit_transform(Z)

	# 释放 GPU 内存缓存，防止内存泄漏，尤其是在多次调用该函数时，确保 GPU 内存得到有效管理。
	torch.cuda.empty_cache()

	# 返回更新后的标签分布
	return Z
