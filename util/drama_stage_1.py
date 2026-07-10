import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat, savemat
from sklearn.utils import shuffle
from build_label_manifold import build_label_manifold
from build_label_manifold2 import build_label_manifold2
import time
# 代码说明
# 构建输入和输出文件名：使用字符串格式化生成输入和输出文件名。
# 加载 .mat 文件：使用 scipy.io.loadmat 加载 .mat 文件中的数据。
# 数据打乱：使用 sklearn.util.shuffle 对特征矩阵和标签矩阵进行打乱。
# 划分训练集和测试集：根据指定的比例（80% 训练集，20% 测试集）划分数据。
# 参数设置：定义各种参数，包括容差、距离阈值、核函数类型、惩罚参数等。
# 调用 build_label_manifold 函数：假设你已经有了一个实现好的 build_label_manifold 函数，并传入训练数据和候选标签矩阵以及近邻数量。
# 保存结果：使用 scipy.io.savemat 将结果保存到 .mat 文件中。
# 注意事项
# build_label_manifold 函数：你需要自己实现这个函数，或者将其替换为你之前已经实现的版本。
# 随机种子：为了保证结果的可重复性，可以在打乱数据时设置随机种子（如 random_state=42）。
# 文件路径：请确保输入文件存在并且路径正确。
# 如果你有更多具体的数值问题或应用场景，欢迎提供更多细节，我可以进一步帮助分析和解决。

def drama_stage_1(target):
    # 2. 构建输入和输出文件名
    input_file = f"{target}.mat"
    output_file = f"{target}_conf_py.mat"

    # 3. 加载 .mat 文件
    data = loadmat(input_file)
    X = data['X']
    Y = data['Y']
    Y_P = data['Y_P']

    # 4. 数据打乱
    m, n = X.shape
    X, Y, Y_P = shuffle(X, Y, Y_P, random_state=42)

    # 5. 划分训练集和测试集
    part = int(m * 0.8)
    train_data = X[:part]
    train_target = Y[:part]
    train_p_target = Y_P[:part]

    test_data = X[part:]
    test_target = Y[part:]

    # 6. 参数设置
    tol = 1e-10
    epsi = 0.1
    ker = 'rbf'
    C1 = 1
    C2 = 0
    k = 10
    par = np.mean(pdist(train_data))

    # 7. 训练阶段
    conf = build_label_manifold(train_data, train_p_target, k)
    #conf = build_label_manifold2(train_data, train_p_target, k)

    # 保存结果到 .mat 文件
    savemat(output_file, {
        'train_data': train_data,
        'train_p_target': train_p_target,
        'train_target': train_target,
        'test_data': test_data,
        'test_target': test_target,
        'conf': conf
    })

if __name__ == "__main__":
    time_s = time.time()
    # 示例用法
    #target = "baselines/DEMIPL_bak/data/emotion/emotion_4"  # 替换为你的目标文件名
    target = "../data/emotion/emotion_4"  # 替换为你的目标文件名
    drama_stage_1(target)
