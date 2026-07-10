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

def build_label_manifold(train_data, train_p_target, k):
    # 2. 初始化变量
    # 功能：获取 train_p_target 的维度，p 表示样本数量，q 表示标签数量。  k是近邻数，d是特征维度
    p, q = train_p_target.shape

    # 3. 归一化特征数据
    # normr 函数：将每一行（即每个样本）的特征向量归一化为单位向量。归一化的作用是确保不同样本之间的距离计算更加公平，避免某些特征值过大或过小影响距离度量。
    train_data = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)

    # 4. 构建 k-d 树并查找最近邻
    # KDTree：构建 k-d 树（k-dimensional tree），用于快速查找最近邻。k-d 树是一种空间分割树，能够高效地进行最近邻搜索。
    kdtree = KDTree(train_data)
    # kdtree.query：在 k-d 树中查找每个样本的 k+1 个最近邻。由于每个样本本身也是自己的最近邻，因此我们只保留 k 个最近邻（去掉第一个最近邻，即样本本身）。
    neighbor_value, neighbor_index = kdtree.query(train_data, k=k + 1)
    # neighbor：存储每个样本的 k 个最近邻的索引，形状为 (p, k)。 所有样本的 第二个开始到第K+1个位置
    neighbor_index = neighbor_index[:, 1:k + 1]  # 去掉第一个最近邻（样本本身）

    # 5. 构建局部线性嵌入（LLE）权重矩阵 W
    # W：局部线性嵌入（Locally Linear Embedding, LLE）权重矩阵，形状为 (p, k)，表示每个样本与其 k 个最近邻之间的权重。
    W = np.zeros((p, k))
    for i in range(p):
        # neighborIns：提取第 i 个样本的 k 个最近邻的特征向量(k,d)[后面的:代表所有特征d]，并通过'符号将其转置为 (d, k) 形状。
        neighborIns = train_data[neighbor_index[i]].T
        # w, _ = lsqnonneg(neighborIns, train_data[i])
        # 求解非负最小二乘问题，找到一组非负权重 w，使得 neighborIns * w ≈ train_data(i,:)，即第 i 个样本可以由其 k 个最近邻的线性组合近似表示。
        w, _ = nnls(neighborIns, train_data[i])
        # W(i, :) = w'：将求得的权重 w 存储到 W 的第 i 行。（k,）可以作为列向量也可以做为行向量
        W[i, :] = w

    # 6. 归一化权重矩阵 W
    # sumW：计算每行权重的和，形状为 (p, 1)【沿第二维sum，维度是从1开始计，第二位就是一行中每列元素sum】。
    sumW = W.sum(axis=1, keepdims=True)
    # sumW(sumW == 0) = 1：为了避免除以零的情况，将和为零的行设置为 1【设置非0的项为1】。
    sumW[sumW == 0] = 1
    # W = W / sumW：将每行权重归一化，使得每行权重之和为 1。这一步确保了局部线性嵌入的权重是概率分布。
    # W = W / sumW
    # 使用 numpy.tile 替代 MATLAB 的 repmat
    repeated_sumW = np.tile(sumW, (1, W.shape[1]))
    # 进行元素级别的除法
    W_normalized = W / repeated_sumW
    # print("原始矩阵 W：")
    # print(W)
    # print("\n每行权重的和 sumW：")
    # print(sumW)
    # print("\n归一化后的矩阵 W_normalized：")
    # print(W_normalized)
    W = W_normalized

    # 7. 构建标签流形矩阵 M
    # M：标签流形矩阵，形状为 (p, p)，用于捕捉样本之间的相似性和标签之间的相关性。
    # sparse：创建一个稀疏矩阵 M，初始值为对角线上的 1，其余元素为 0。稀疏矩阵的使用可以节省内存并提高计算效率。
    M = csr_matrix((np.ones(p), (np.arange(p), np.arange(p))), shape=(p, p))
    # for ii = 1:p：遍历每个样本 ii，更新 M 矩阵的对应行和列。
    for ii in range(p):
        # w = W(ii, :)：提取第 ii 个样本的 k 个最近邻的权重
        w = W[ii]
        # jj = neighbor(ii, :)：提取第 ii 个样本的 k 个最近邻的索引。
        jj = neighbor_index[ii]
        # M(ii, jj) = M(ii, jj) - w：更新 M 矩阵中第 ii 行与 jj 列的元素，减去对应的权重。
        # 这里 M[ii, jj] 是一个稀疏矩阵的子集，形状为 (1, k)。
        # w 的形状是 (k,)，可以直接广播到 (1, k)。
        M[ii, jj] -= w
        # M(jj, ii) = M(jj, ii) - w'：更新 M 矩阵中第 jj 行与 ii 列的元素，减去对应的权重。
        # M[jj, ii] 的形状是 (k, 1)。
        # w[:, None] 将 w 的形状从 (k,) 转换为 (k, 1)，使其与 M[jj, ii] 的形状匹配。
        M[jj, ii] -= w[:, None]  # 将 w 转换为 (k, 1) 形状
        #  M(jj, jj) = M(jj, jj) + w' * w：更新 M 矩阵中第 jj 行与 jj 列的元素，加上权重的外积。这一步确保了流形矩阵的对称性和正定性。
        # np.outer(w, w) 计算 w 的外积，形状为 (k, k)。
        # jj[:, None] 将 jj 的形状从 (k,) 转换为 (k, 1)，使其与 jj 的形状匹配。
        M[jj[:, None], jj] += np.outer(w, w)
        # M = M - csr_matrix((w, (np.full(k, ii), jj)), shape=(p, p))
        # M = M - csr_matrix((w, (jj, np.full(k, ii))), shape=(p, p))
        # M = M + csr_matrix((np.outer(w, w).flatten(), (np.repeat(jj, k), np.tile(jj, k))), shape=(p, p))

    # 8. 扩展标签流形矩阵 M1 【形成 M1(p*q,p*q)，里面是被一块（q*q）小块区域，对角线上是当前p_i标签对应别的标签的相关性】
    # kron：计算 Kronecker 积，将 M 矩阵扩展为 q 倍大小的矩阵 M1，形状为 (p*q, p*q)。Kronecker 积的作用是将每个标签视为独立的空间，同时保持样本之间的相似性关系。
    # eye(q)：生成一个 q x q 的单位矩阵，表示每个标签的空间是独立的。【空间扩大了】
    # M1：扩展后的标签流形矩阵，形状为 (p*q, p*q)，用于捕捉标签之间的相关性和样本之间的相似性。
    M_extend_reg = M + 1e-6 * np.eye(M.shape[0])  # 强制正则化
    M1 = kron(M_extend_reg, eye(q)).tocsr()

    # 9. 设置边界条件
    # lb 和 ub：分别为下界和上界，形状为 (p*q, 1)，用于约束优化问题中的变量范围，其中p*q是长度，先设置为无限大和无限小。
    num = p * q
    #lb = np.full(num, -np.inf)
    #ub = np.full(num, np.inf)
    lb = np.full(num, -10000)  # 不能设置inf，太大模型会爆，所以缩小8个0
    ub = np.full(num, 10000)

    # label：将 train_p_target 展平为一维向量，形状为 (p*q, 1)，表示每个标签的候选置信度。
    # 先把标签重构成长序列  (p*q ,1)
    label = train_p_target.flatten()
    # delta1 和 delta2：分别表示置信度为 1 和 0 的标签的边界调整参数。【上限和下限设置好】
    # -0.1, -0.01, 0, 1
    delta1 = 0.01
    # -1 -0.5 0
    delta2 = 0.5

    # % for i = 1:num：遍历每个标签，根据其候选置信度设置相应的上下界：
    for i in range(num):
        # # 如果标签的候选置信度为 1，则将其下界设置为 -delta1，表示该标签的置信度不能太低。
        # if label[i] == 1:
        #     lb[i] = -delta1
        # #  如果标签的候选置信度为 0，则将其上界设置为 -delta2，表示该标签的置信度不能太高
        # elif label[i] == 0:
        #     ub[i] = -delta2
        # 如果标签的候选置信度为 1，则将其下界设置为 -delta1，表示该标签的置信度不能太低。
        if label[i] > delta1:
            lb[i] = -delta1
        #  如果标签的候选置信度为 0，则将其上界设置为 -delta2，表示该标签的置信度不能太高
        elif label[i] <= 0:
            ub[i] = -delta2

    # 10. 构建约束矩阵 A 和向量 b 【把标签带约束填回A，用于后续优化计算】
    # scipy.sparse.csr_matrix 是一种用于表示稀疏矩阵的数据结构。稀疏矩阵是指大部分元素为零的矩阵，使用这种数据结构可以显著减少内存占用和计算时间。
    # 这个构造函数实际上是创建一个形状为 (p, num) 的空稀疏矩阵（即所有元素初始值为零）。
    #  A：约束矩阵，形状为 (p（样本）, p*q(样本*标签))，用于表示每个样本的标签约束。
    A = csr_matrix((p, num))
    for kkk in range(p):
        # for kkk = 1:p：遍历每个样本，将 train_p_target 中的候选置信度填入 A 矩阵的相应位置。A 的每一行对应一个样本，每一列对应一个标签[A=行样本，列标签（q个标签，稀疏的kkk * q:(kkk + 1) * q的位置）]。
        A[kkk, kkk * q:(kkk + 1) * q] = train_p_target[kkk]
    # b：约束向量，形状为 (p, 1)，表示每个样本的标签总和应为 1。 【每个样本有个b的偏置项】
    b = np.ones(p)

    # 11. 设置优化选项
    # solvers.options['show_progress'] = True
    # solvers.options['abstol'] = 1e-7
    # solvers.options['reltol'] = 1e-7
    # solvers.options['feastol'] = 1e-7

    # 将矩阵和向量转换为 cvxopt 矩阵格式（密集矩阵）
    M1_cvx = csr_matrix(M1.toarray())  # 将 M1 转换为稀疏矩阵格式
    A_cvx = csr_matrix(A.toarray())  # 将 A 转换为稀疏矩阵格式
    b_cvx = np.array(b)  # 将 b 转换为 numpy 数组
    lb_cvx = np.array(lb)  # 将 lb 转换为 numpy 数组
    ub_cvx = np.array(ub)  # 将 ub 转换为 numpy 数组

    # 12. 求解二次规划问题
    H = 2 * M1_cvx  # H 是二次项系数矩阵
    H_reg = H + 1e-6 * np.eye(H.shape[0])  # 正则化
    #H_reg = H_reg / np.linalg.norm(H)  # 对 H 进行归一化：
    f = csr_matrix(np.zeros(num))  # f 是线性项系数向量

    # 12（2） 求解二次规划问题（手动重写）
    x = cp.Variable(num)  # 定义了优化变量 x，维度为 num，并且 pos=True 表示 x 的每个元素都是非负的。
    # 目标函数， min 1/2 x^{T}(2M1)x
    obj = cp.Minimize(0.5 * cp.quad_form(x, H_reg))
    # 同时满足以下约束条件：
    # 1.标签总和约束：每个样本的标签总和应为 1，即A*x=b。
    # 2.边界约束：每个标签的置信度应在设定的范围内，即lb≤x≤ub。
    con = [A_cvx @ x >= b_cvx, lb_cvx <= x, x <= ub_cvx]
    # con2 = [lb_cvx <= x, x <= ub_cvx]
    con3 = [A_cvx @ x >= b_cvx]  # 可行
    # con4 = []
    # con5 = [A_cvx @ x >= b_cvx, lb_cvx <= x, x <= ub_cvx]
    prob = cp.Problem(obj, con)


    # 过程诊断
    cond_number = np.linalg.cond(H.toarray())
    # 检查 H 是否是正定矩阵。可以通过计算其特征值来验证：
    eigenvalues = np.linalg.eigvals(H.toarray())
    # 如果存在非正特征值，可以尝试对 H 进行正则化：
    #H = H + 1e-6 * np.eye(H.shape[0])
    print("H 的特征值：", eigenvalues)
    print("H 的条件数：", cond_number)
    print("H 的形状：", H.shape)
    print("x 的形状：", x.shape)
    print("A_cvx 的形状：", A_cvx.shape)
    print("b_cvx 的形状：", b_cvx.shape)
    print("lb_cvx 的形状：", lb_cvx.shape)
    print("ub_cvx 的形状：", ub_cvx.shape)

    #  二次规划计算
    prob.solve(solver=cp.CVXOPT, verbose=True)
    #prob.solve(solver='ECOS', verbose=True)
    #prob.solve(solver=cp.CVXOPT, verbose=True, eps=1e-1, max_iters=100000)
    #  x.value 是优化变量的最优解。
    # prob.value 是目标函数的最优值。
    # 13. 输出结果
    if prob.status == cp.OPTIMAL:
        print('最优解为：', np.round(x.value, 4))
        print('最优值为：', np.round(prob.value, 4))
    else:
        print('求解失败，状态：', prob.status)

    # 创建问题并求解
    # try:
    #
    #     # 尝试使用 CVXOPT 求解器
    #     print("尝试使用 CVXOPT 求解器...")
    #     prob.solve(solver=cp.CVXOPT, verbose=True)
    # except cp.error.SolverError as e:
    #     print(f"CVXOPT 求解失败: {e}")
    #
    #     try:
    #         # 如果 CVXOPT 失败，尝试使用 ECOS 求解器
    #         print("尝试使用 ECOS 求解器...")
    #         prob.solve(solver=cp.ECOS, verbose=True)
    #     except cp.error.SolverError as e:
    #         print(f"ECOS 求解失败: {e}")
    #
    #         try:
    #             # 如果 ECOS 也失败，尝试使用 SCS 求解器
    #             print("尝试使用 SCS 求解器...")
    #             prob.solve(solver=cp.SCS, verbose=True)
    #         except cp.error.SolverError as e:
    #             print(f"SCS 求解失败: {e}")
    #             raise Exception("所有求解器均失败，请检查输入数据和约束条件。")

    # 获取最优解
    optimal_solution = x.value

    # 输出结果
    if optimal_solution is not None:
        print("最优解为：", np.round(optimal_solution, 4))
        print("最优值为：", np.round(prob.value, 4))
    else:
        print("未找到可行解。")


    #  13. 重塑输出结果
    Outputs = np.array(x.value).reshape(q, p).T
    print(Outputs)

    return Outputs




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