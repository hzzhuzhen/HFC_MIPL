import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import nnls
#from scipy.optimize import lsqnonneg
from scipy.sparse import csr_matrix, kron, eye
from cvxopt import matrix, solvers
import cvxpy as cp
from scipy.optimize import minimize
#######################
# 代码说明
# 归一化特征数据：使用 np.linalg.norm 对每一行进行归一化。
# 构建 k-d 树并查找最近邻：使用 scipy.spatial.KDTree 构建 k-d 树，并使用 query 方法查找最近邻。
# 构建局部线性嵌入（LLE）权重矩阵 W：使用 scipy.optimize.lsqnonneg 求解非负最小二乘问题。
# 归一化权重矩阵 W：对每行权重进行归一化处理。
# 构建标签流形矩阵 M：使用 scipy.sparse.csr_matrix 创建稀疏矩阵，并对其进行更新。
# 扩展标签流形矩阵 M1：使用 scipy.sparse.kron 计算 Kronecker 积。
# 设置边界条件：根据标签置信度设置上下界。
# 构建约束矩阵 A 和向量 b：创建稀疏矩阵并填充相应值。
# 求解二次规划问题：使用 cvxopt.solvers.qp 求解二次规划问题。
# 重塑输出结果：将结果重新组织为二维矩阵。
# 这个 Python 版本的代码尽可能地保留了原始 MATLAB 代码的功能和逻辑结构。如果有任何特定的需求或进一步的问题，请随时告诉我！
#
#######################

def build_label_manifold2(train_data, train_p_target, k):
    # 2. 初始化变量
    p, q = train_p_target.shape

    # 3. 归一化特征数据
    train_data = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)

    # 4. 构建 k-d 树并查找最近邻
    kdtree = KDTree(train_data)
    neighbor_value, neighbor_index = kdtree.query(train_data, k=k + 1)
    neighbor_index = neighbor_index[:, 1:k + 1]  # 去掉第一个最近邻（样本本身）

    # 5. 构建局部线性嵌入（LLE）权重矩阵 W
    W = np.zeros((p, k))
    for i in range(p):
        neighborIns = train_data[neighbor_index[i]].T
        # w, _ = lsqnonneg(neighborIns, train_data[i])
        w, _ = nnls(neighborIns, train_data[i])
        W[i, :] = w

    # 6. 归一化权重矩阵 W
    sumW = W.sum(axis=1, keepdims=True)
    sumW[sumW == 0] = 1
    W =W / sumW

    # 7. 构建标签流形矩阵 M
    M = csr_matrix((np.ones(p), (np.arange(p), np.arange(p))), shape=(p, p))
    for ii in range(p):
        w = W[ii]
        jj = neighbor_index[ii]
        # 这里 M[ii, jj] 是一个稀疏矩阵的子集，形状为 (1, k)。
        # w 的形状是 (k,)，可以直接广播到 (1, k)。
        M[ii, jj] -= w
        # M[jj, ii] 的形状是 (k, 1)。
        # w[:, None] 将 w 的形状从 (k,) 转换为 (k, 1)，使其与 M[jj, ii] 的形状匹配。
        M[jj, ii] -= w[:, None]  # 将 w 转换为 (k, 1) 形状
        # np.outer(w, w) 计算 w 的外积，形状为 (k, k)。
        # jj[:, None] 将 jj 的形状从 (k,) 转换为 (k, 1)，使其与 jj 的形状匹配。
        M[jj[:, None], jj] += np.outer(w, w)
        # M = M - csr_matrix((w, (np.full(k, ii), jj)), shape=(p, p))
        # M = M - csr_matrix((w, (jj, np.full(k, ii))), shape=(p, p))
        # M = M + csr_matrix((np.outer(w, w).flatten(), (np.repeat(jj, k), np.tile(jj, k))), shape=(p, p))

    # 8. 扩展标签流形矩阵 M1
    M_extend_reg = M + 1e-6 * np.eye(M.shape[0])  # 强制正则化
    M1 = kron(M_extend_reg, eye(q)).tocsr()

    # 9. 设置边界条件
    num = p * q
    lb = np.full(num, -np.inf)
    ub = np.full(num, np.inf)
    label = train_p_target.flatten()
    delta1 = 0.01
    delta2 = 0.5

    for i in range(num):
        if label[i] == 1:
            lb[i] = -delta1
        elif label[i] == 0:
            ub[i] = -delta2

    # 10. 构建约束矩阵 A 和向量 b
    # scipy.sparse.csr_matrix 是一种用于表示稀疏矩阵的数据结构。稀疏矩阵是指大部分元素为零的矩阵，使用这种数据结构可以显著减少内存占用和计算时间。
    # 这个构造函数实际上是创建一个形状为 (p, num) 的空稀疏矩阵（即所有元素初始值为零）。
    A = csr_matrix((p, num))
    for kkk in range(p):
        A[kkk, kkk * q:(kkk + 1) * q] = train_p_target[kkk]
    b = np.ones(p)

    # 11. 设置优化选项
    # solvers.options['show_progress'] = False
    # solvers.options['abstol'] = 1e-7
    # solvers.options['reltol'] = 1e-7
    # solvers.options['feastol'] = 1e-7

    # 将矩阵和向量转换为 cvxopt 矩阵格式（密集矩阵）
    # 将稀疏矩阵转换为密集矩阵
    M1_dense = M1.toarray()
    A_dense = A.toarray()
    M1_cvx = csr_matrix(M1_dense)  # 将 M1 转换为稀疏矩阵格式
    A_cvx = csr_matrix(A_dense)  # 将 A 转换为稀疏矩阵格式
    b_cvx = np.array(b)  # 将 b 转换为 numpy 数组
    lb_cvx = np.array(lb)  # 将 lb 转换为 numpy 数组
    ub_cvx = np.array(ub)  # 将 ub 转换为 numpy 数组



    H = 2 * M1_cvx  # H 是二次项系数矩阵
    H_reg = H + 1e-6 * np.eye(H.shape[0])  # 正则化
    #H_reg = H_reg / np.linalg.norm(H)  # 对 H 进行归一化：
    f = csr_matrix(np.zeros(num))  # f 是线性项系数向量

    # 12（2） 求解二次规划问题（手动重写）
    #x = cp.Variable(num, pos=True)  # 定义了优化变量 x，维度为 num，并且 pos=True 表示 x 的每个元素都是非负的。
    # 目标函数， min 1/2 x^{T}(2M1)x
    #obj = cp.Minimize(0.5 * cp.quad_form(x, H_reg))
    # 同时满足以下约束条件：
    # 1.标签总和约束：每个样本的标签总和应为 1，即A*x=b。
    # 2.边界约束：每个标签的置信度应在设定的范围内，即lb≤x≤ub。
    #con = [A_cvx @ x == b_cvx, lb_cvx <= x, ub_cvx >= x]
    #prob = cp.Problem(obj, con)

    # 过程诊断
    cond_number = np.linalg.cond(H.toarray())
    # 检查 H 是否是正定矩阵。可以通过计算其特征值来验证：
    eigenvalues = np.linalg.eigvals(H.toarray())
    # 如果存在非正特征值，可以尝试对 H 进行正则化：
    #H = H + 1e-6 * np.eye(H.shape[0])
    print("H 的特征值：", eigenvalues)
    print("H 的条件数：", cond_number)
    print("H 的形状：", H.shape)
    #print("x 的形状：", x.shape)
    print("A_cvx 的形状：", A_cvx.shape)
    print("b_cvx 的形状：", b_cvx.shape)
    print("lb_cvx 的形状：", lb_cvx.shape)
    print("ub_cvx 的形状：", ub_cvx.shape)

    # 12. 求解二次规划问题
    # 定义目标函数
    def objective(x):
        return 0.5 * np.dot(np.dot(x.T, 2 * M1_dense), x)

    # 定义不等式约束
    def inequality_constraints(x):
        return -np.dot(A_dense, x) + b

    # 定义边界约束
    bounds = [(lb[i], ub[i]) for i in range(num)]

    # 初始猜测值
    x0 = np.zeros(num)

    # 定义约束条件
    constraints = ({'type': 'ineq', 'fun': inequality_constraints})
    # 使用 SLSQP 方法进行优化
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    # 获取最优解
    Outputs = result.x
    # 13. 重塑输出结果
    Outputs_reshaped = Outputs.reshape(q, p).T

    print(Outputs_reshaped)

    return Outputs


# # 示例用法
# train_data = np.random.rand(10, 5)  # 示例训练数据
# train_p_target = np.random.randint(0, 2, size=(10, 3))  # 示例标签目标
# k = 2  # 最近邻数量
#
# result = build_label_manifold(train_data, train_p_target, k)
# print(result)


# % 附录4：二次规划（Quadratic Programming, QP） 求解器来优化标签的置信度，并将优化后的结果重新组织为二维矩阵。
# % 1. 设置优化选项
# % options = optimoptions('quadprog', ...
# %    'Display', 'off', 'Algorithm', 'interior-point-convex');
# % optimoptions 函数：用于设置优化求解器的选项，特别是针对 quadprog 函数的二次规划求解器。
# % 'Display', 'off'：关闭求解过程中的输出显示，避免不必要的信息输出。这可以提高程序的运行效率，尤其是在处理大规模数据集时，避免过多的中间输出信息干扰用户。
# % 'Algorithm', 'interior-point-convex'：选择凸二次规划的 内点法算法（Interior-Point Convex Algorithm）。内点法是一种高效的凸优化算法，适用于解决凸二次规划问题。凸优化问题具有全局最优解的性质，因此选择内点法可以确保找到全局最优解。
# % options：保存了上述设置的优化选项，后续将传递给 quadprog 函数。
#
# % 内点法算法的作用
# % 内点法算法是一种迭代优化方法，它通过在可行域内部寻找最优解，逐步逼近目标函数的最小值。对于凸二次规划问题，内点法能够保证找到全局最优解，并且在大多数情况下具有较快的收敛速度。选择该算法的原因是：
# % 凸优化问题：二次规划问题中，如果目标函数和约束条件都是凸的，则问题具有全局最优解。
# % 高效性：内点法在处理大规模凸优化问题时表现出色，能够在较短时间内找到最优解。
# % 2. 定义约束向量 b
# % b = ones(p, 1);
# % ones(p, 1)：创建一个形状为 (p, 1) 的全 1 向量 b，其中 p 是样本数量。
# % 含义：b 用于表示每个样本的标签总和应为 1 的约束条件。具体来说，b(i) = 1 表示第 i 个样本的标签总和应为 1，即每个样本的标签置信度之和为 1。
# % 3. 求解二次规划问题
# % Outputs = quadprog(2*M1, [], -A, -b, [], [], lb, ub, [], options);
# % quadprog 函数：用于求解 二次规划问题，其标准形式为：
# % min_x 1/2 x^{T}Hx+f^{T}x
# % 其中：
# % H 是对称正定矩阵，表示目标函数中的二次项。
# % f 是线性项的系数向量。x是待优化的变量向量。
# % 2*M1：扩展后的标签流形矩阵 M1，形状为 (p*q, p*q)。2*M1 作为二次项矩阵
# % H，表示目标函数中的二次项。乘以 2 是因为 quadprog 的目标函数中有1/2的系数，为了保持等价性，通常将 H 乘以 2。
# % []：表示没有线性项f，即目标函数中没有线性项。
# % -A 和 -b：表示不等式约束−A*x≤−b，即A*x≥b。这里 A 是约束矩阵，b 是约束向量。
# % A 的每一行对应一个样本，b 的每个元素为 1，表示每个样本的标签总和应为 1。
# % lb 和 ub：分别为下界和上界，形状为 (p*q, 1)，用于约束优化问题中的变量范围。lb 和 ub 的设置确保每个标签的置信度在合理的范围内。
# % options：传递给 quadprog 的优化选项，包括关闭输出显示和选择内点法算法。
# % Outputs：求解得到的最优标签置信度向量，形状为 (p*q, 1)，表示每个标签-样本对的优化后置信度。
# % 二次规划问题的目标
# % 二次规划问题的目标是最小化以下目标函数：
# % min 1/2 x^{T}(2M1)x
# % 同时满足以下约束条件：
# % 1.标签总和约束：每个样本的标签总和应为 1，即A*x=b。
# % 2.边界约束：每个标签的置信度应在设定的范围内，即lb≤x≤ub。
# % 通过求解这个二次规划问题，我们可以在满足约束条件的前提下，找到使得目标函数最小化的最优标签置信度。
# % 4. 重塑输出结果
# % Outputs = reshape(Outputs, q, p)';
# % reshape(Outputs, q, p)：将一维向量 Outputs 重新组织为二维矩阵，形状为 (q, p)。q 是标签数量，p 是样本数量。
# % '：转置操作，将 (q, p) 形状的矩阵转置为 (p, q) 形状。
# % 含义：经过 quadprog 求解后，Outputs 是一个形状为 (p*q, 1) 的一维向量，表示每个标签-样本对的优化后置信度。通过 reshape 操作，我们将这个一维向量重新组织为二维矩阵，形状为 (p, q)，表示每个样本对每个标签的优化后置信度。
# % 重塑的意义
# % (p, q) 形状的矩阵：重塑后的 Outputs 矩阵的每一行对应一个样本，每一列对应一个标签。这样可以方便地查看每个样本的标签置信度分布，并与其他部分的代码进行进一步处理或分析。
# % 5. 结束函数
# % end
# % end：表示函数的结束。
# % 总结
# % options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex')：设置二次规划求解器的选项，关闭输出显示并选择内点法算法，确保求解凸二次规划问题。
# % b = ones(p, 1)：定义约束向量 b，表示每个样本的标签总和应为 1。
# % Outputs = quadprog(2*M1, [], -A, -b, [], [], lb, ub, [], options)：求解二次规划问题，目标是最小化 0.5 * x' * H * x，其中 H = 2*M1 是扩展后的标签流形矩阵，f 为空（即无线性项）。约束条件包括标签总和约束和边界约束。Outputs 是求解得到的最优标签置信度向量，形状为 (p*q, 1)。
# % Outputs = reshape(Outputs, q, p)'：将一维向量 Outputs 重新组织为二维矩阵，形状为 (p, q)，表示每个样本对每个标签的优化后置信度。