#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import argparse  
import os  
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
grandparent_path = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, grandparent_path)
sys.path.insert(1, parent_path)
sys.path.insert(2, current_path)
print('current dir:', sys.path[0])
import time  
import torch  
import torch.utils.data as data_utils  
import torch.optim as optim  
from torch.autograd import Variable 
from sklearn.metrics import accuracy_score  
from scipy.special import softmax

from HFC_MIPL.model_hfc import *  
from utils import *  
from HFC_MIPL.graph.deep_manifold_hfc import *
from utils import *
from dataloader import * 

# mix-loss:CrossEntropyLoss: 
#from weight_loss import CrossEntropyLoss as CE


# training settings
parser = argparse.ArgumentParser(
    description='Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R', help='weight decay')
parser.add_argument('--w_entropy_A', type=float, default=0.001, metavar='L', help='weight of the loss function')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
parser.add_argument('--index', type=str, default='index', help='index path')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='MNIST_MIPL, FMNIST_MIPL, ...')
parser.add_argument('--ds_suffix', type=str, default= None, help='the specific type of the data set')
parser.add_argument('--bs_tr', type=int, default=1, help='batch size for training ')
parser.add_argument('--bs_te', type=int, default=1, help='batch size for testing ')
parser.add_argument('--nr_fea', type=int, default=784, help='feature dimension of an instance ')
parser.add_argument('--nr_class', type=int, default=5, help='classes of bag ')
parser.add_argument('--normalize', type=str2bool, default=False, help='normalize the dataset, True or False ')

# manifold parameter
parser.add_argument('--using_lp', action='store_true')
parser.add_argument('--using_prep', action='store_true')
parser.add_argument('--class_cons', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--hidden_size', type=str, default='64,64')
parser.add_argument('--gpuid', type=int, default=0)
# dualg parameters
parser.add_argument('--tr_rate', type=float, default=0.9)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--gamma', type=float, default=10., metavar='L', help='the weight of the inhibition loss')
parser.add_argument('--maxiter', type=int, default=1)
parser.add_argument('--neighbors_num', type=int, default=10)
parser.add_argument('--no-verbose', action='store_true')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.00)
parser.add_argument('--lambda_recon_rate', type=float, default=0.1, metavar='L', help='graph bag feature manifold')
parser.add_argument('--mu', type=float, default=0.2, metavar='L', help='the weight of the sparsity loss')
parser.add_argument('--mu2', type=float, default=0.05, metavar='L', help='the weight of the sparsity loss')
parser.add_argument('--L', type=int, default=500, metavar='S', help='the embedded space dimension ')
parser.add_argument('--la', type=float, default=2.0, metavar='L', help='residual rate-discarded')
parser.add_argument('--mix_loss_rate', type=float, default=0.1, metavar='L', help='mixloss rate')
parser.add_argument('--isMixLoss', type=str2bool, default=False, help='open mixloss，True or False')
parser.add_argument('--current_epoch', type=int, default=0, metavar='S', help='current epoch')


args = parser.parse_args()
print("whole parameters：")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# seed
seed_everything(args.seed)

# gpu available
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.cuda:
    print('\nGPU is available!')

# data index
all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
             'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']

#  evaluate 
def evaluate(loader, model):
    '''
    model testing
    '''
    model.eval()

    all_true_bag_lab = []
    all_pred_bag_lab = []
    all_pred_bag_prob = np.empty((0, args.nr_class))
    for data, _, true_bag_lab, _ in loader:
        data = data.to(device)
        true_bag_lab = true_bag_lab.to(device)
        data = data.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # evaluate model
        output = model.evaluate_objective(data, args)
        # processing result
        all_pred_bag_prob = np.vstack((all_pred_bag_prob, output.detach().cpu().numpy()))
        _, pred_bag_lab = torch.max(output.data, 1)
        all_true_bag_lab.append(true_bag_lab.item())
        all_pred_bag_lab.append(pred_bag_lab.item())
    # cal result
    all_true_bag_lab = np.array(all_true_bag_lab)
    all_pred_bag_lab = np.array(all_pred_bag_lab)
    acc = accuracy_score(all_true_bag_lab, all_pred_bag_lab)
    return acc


# train method
def train_deepm(epoch,X_recon=None, old_aggre_bag_data_list = None, Y_P_train=None, Y_recon=None,lambda_recon_rate=0.01, model=None, optimizer=None):
    '''
     model training 
     X_recon,aggre_bag_data_list,Y_pred_np, Y_P_train,Y_recon
    '''

    model.train()
    train_loss = 0.
    attention_score_np = np.empty((0, 1))

    attention_score_list = []
    aggre_bag_data_list = []
    f_list = []
    partial_bag_lab_list = []
    true_bag_lab_list = []
    for batch_idx, (data, partial_bag_lab, true_bag_lab, index) in enumerate(train_loader):
        # data preprocessing
        if args.cuda:
            data, partial_bag_lab, true_bag_lab = data.cuda(), partial_bag_lab.cuda(), true_bag_lab.cuda()
        data, partial_bag_lab, true_bag_lab = Variable(data), Variable(partial_bag_lab), Variable(true_bag_lab)
        data = data.to(torch.float32)
        partial_bag_lab = partial_bag_lab.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        X_recon_iter = None
        aggre_data_iter = None
        Y_P_train_iter = None
        Y_recon_iter = None

        if (X_recon is not None) and (Y_P_train is not None) and (Y_recon is not None) and (old_aggre_bag_data_list is not None):

            X_recon_iter = X_recon[index]
            aggre_data_iter = old_aggre_bag_data_list[index]
            Y_P_train_iter = Y_P_train[index]
            Y_recon_iter = Y_recon[index]

        loss, new_partial_bag_lab, attention_score,f = model.calculate_objective_hfc_deepm(args,data,partial_bag_lab, 
                                                                                     X_recon_iter,aggre_data_iter,Y_P_train_iter,Y_recon_iter)
                                                                                     

        # 3. model post processing
        attention_score_copy = np.copy(attention_score.detach().cpu().numpy())
        # softmax
        attention_score_copy = softmax(attention_score_copy, axis=1)
        attention_score_list.insert(batch_idx, attention_score_copy)
        # aggregation data
        data_shape = data.shape
        agrea_bag_data = torch.from_numpy(attention_score_copy).to(device).view(-1) @ data.view(data_shape[0] * data_shape[1], -1)
        aggre_bag_data_list.insert(batch_idx, np.copy(agrea_bag_data.detach().cpu().numpy()))
        f_list.insert(batch_idx, np.copy(f.detach().cpu().numpy()))

        train_loss += loss.item()
        lamda = lambda_list[epoch - 1]
        new_partial_bag_lab = new_partial_bag_lab.cpu().detach().numpy()
        partial_bag_lab = partial_bag_lab.cpu().detach().numpy()
        new_label = lamda * partial_bag_lab + (1. - lamda) * new_partial_bag_lab
        new_label = np.squeeze(new_label, axis=0)
        train_loader.dataset.train_partial_bag_lab_list[index] = new_label
        # backward pass
        loss.backward()
        optimizer.step()
        partial_bag_lab_list.insert(batch_idx, np.copy(np.squeeze(new_label, axis=0)))  # 先用原本的 new_label已经更新了数据
        true_bag_lab_np = true_bag_lab.cpu().numpy()
        true_bag_lab_list.insert(batch_idx, np.copy(true_bag_lab_np))
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    if epoch == 1 or (epoch) % 10 == 0:
        print('Epoch: {}, Train loss: {:.4f}.'.format(epoch, train_loss))
    if (epoch) % 30 == 1:
        print('Epoch: {}, deepm manifold update label started...'.format(epoch))
        aggre_bag_data_list = np.array(aggre_bag_data_list)
        aggre_bag_data_tensor = torch.from_numpy(aggre_bag_data_list).cpu()
        partial_bag_lab_list = np.array(partial_bag_lab_list)
        partial_bag_lab_tensor = torch.from_numpy(partial_bag_lab_list).cpu()
        f_list=np.array(f_list)
        f_list_tensor = torch.from_numpy(f_list).cpu()

        # manifold model cal
        print("deepm:build_label_manifold....")
        # 5.init label
        if Y_recon is None:
            Y_recon = partial_bag_lab_list
        # 6. bag feature graph building
        # X_recon_aggre =aggre_bag_data_list
        # X_recon = X_recon_aggre
        # X_bag_train = X_recon
        # Y_P_train = partial_bag_lab_list 
        # Y_P_train, Y_recon, Wn, L ,X_recon= deepm_train_global_prepare(args, X_bag_train, Y_P_train)
        
        # # aggre graph builidng V2
        X_f_recon = f_list
        Y_P_train = partial_bag_lab_list 
        Y_P_train, Y_recon, Wn_f, L,X_recon = deepm_train_global_prepare(args, X_f_recon, Y_P_train)
        
    return model, train_loss,X_recon,aggre_bag_data_list, Y_P_train,Y_recon
    #return model, train_loss,X_f_recon,aggre_bag_data_list, Y_P_train,Y_recon
# 实例选择
def adaptive_instance_selection(attention_scores, min_instances=1, max_instances=6, 
                               selection_strategy='hybrid', percentile_threshold=75, 
                               std_multiplier=0.5, relative_ratio=0.7):
    """
    adaptive instance selection，accoridng attention score distribution
    Returns:
        selected_indices
    """
    scores = attention_scores[0]  # shape: [num_instances]
    num_instances = len(scores)
    
    if selection_strategy == 'percentile':
        threshold = np.percentile(scores, percentile_threshold)
        candidate_indices = np.where(scores >= threshold)[0]
        
    elif selection_strategy == 'std':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + std_multiplier * std_score
        candidate_indices = np.where(scores >= threshold)[0]
        
    elif selection_strategy == 'relative':
        max_score = np.max(scores)
        threshold = max_score * relative_ratio
        candidate_indices = np.where(scores >= threshold)[0]
    
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
    
    # score desc
    candidate_scores = scores[candidate_indices]
    sorted_order = np.argsort(candidate_scores)[::-1]
    selected_indices = candidate_indices[sorted_order]
    
    return selected_indices

#     Epoch-adaptive noisy instance filtering 动态实例筛选
def epoch_adaptive_noisy_instance_filtering(
    attention_scores,
    epoch: int,          # 当前训练轮次 t (论文τₜ的t)
    update_interval: int = 10,  # 阈值更新间隔 T
    tau0: float = 0.2,   # 初始阈值 τ₀
    delta: float = 0.02, # 阈值递增步长 Δ
    tau_max: float = 0.6,# 阈值上限 τ_max
    min_instances: int = 1,
    max_instances: int = 6
):
    """
    Epoch-adaptive noisy instance filtering 动态实例筛选
    完全对齐论文公式 τ_t = min(τ₀ + Δ * floor(t / T), τ_max)
    Args:
        attention_scores: 注意力权重，shape [1, num_instances]
        epoch: 当前训练轮次 t
        update_interval T: 每T轮提升一次阈值
        tau0: 训练初始阈值
        delta: 单次阈值提升步长
        tau_max: 阈值最大值，不再持续上升
        min_instances: 每包最少保留实例数量
        max_instances: 每包最多保留实例数量
    Returns:
        np.ndarray: 筛选后高可靠实例下标，按注意力从大到小排序
    """
    scores = attention_scores[0]  # [num_instances]
    num_total = len(scores)
    
    # 论文自适应阈值计算公式
    floor_term = epoch // update_interval
    tau_t = tau0 + delta * floor_term
    tau_t = min(tau_t, tau_max)  # 限制阈值上限
    
    # 筛选注意力高于阈值的候选实例
    candidate_indices = np.where(scores >= tau_t)[0]
    candidate_scores = scores[candidate_indices]
    
    # 按注意力降序排序
    sort_idx = np.argsort(candidate_scores)[::-1]
    selected = candidate_indices[sort_idx]
    
    # 约束保留实例数量 [min, max]
    if len(selected) < min_instances:
        # 候选不足，直接取全局top-min实例兜底
        all_sort = np.argsort(scores)[::-1]
        selected = all_sort[:min_instances]
    elif len(selected) > max_instances:
        # 候选过多，截断至最多实例数
        selected = selected[:max_instances]
    
    return selected
# adjust_lambda fuction
def adjust_lambda(epochs):
    '''
    the momentum parameter in Equation (7)
    '''
    lambda_list = [1.0] * epochs
    for ep in range(epochs):
        lambda_list[ep] = (epochs - ep) / (epochs)

    return lambda_list

# main fuction
if __name__ == "__main__":
    time_s = time.time()
    lambda_list = adjust_lambda(args.epochs)
    num_trial = 1
    num_fold = len(all_folds)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)
    mat_name = args.ds + '.mat'
    if args.ds_suffix is not None:
        mat_name = args.ds + '_r' + args.ds_suffix + '.mat'
    mat_path = os.path.join(data_path, mat_name)
    ds_name = mat_name[0:-4]
    # data loading
    all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
        mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

    accuracy = np.empty((num_trial, num_fold))
    # evaluation
    for trial_i in range(num_trial):
        for fold_i in range(num_fold):
            print('\n---------------- time: %d, fold: %d ----------------' % (trial_i + 1, fold_i + 1))
            idx_file = index_path + '/' + all_folds[fold_i]
            # load the index and dataset
            idx_tr, idx_te = load_idx_mat(idx_file)
            #  MIPLDataloader loading
            train_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=True, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=False, **loader_kwargs)
            test_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=False, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=False, **loader_kwargs)

            # ----------------  init model ----------------
            model = HFCLossGatedAttention(args)
            print("输出模式使用的参数")
            for name, param in model.named_parameters():
                print(name, param.shape)
            if args.cuda:
                model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.reg)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # -------------- start training ---------------
            X_recon = None 
            aggre_bag_data_list = None
            Y_P_train = None 
            Y_recon = None

            for epoch in range(1, args.epochs + 1):
                args.current_epoch = epoch
                model, train_loss,X_recon,aggre_bag_data_list, Y_P_train,Y_recon = train_deepm(epoch,X_recon,aggre_bag_data_list, Y_P_train,Y_recon,model=model,optimizer=optimizer)
                lr_scheduler.step()

            # -------------- start testing ---------------
            test_accuracy = evaluate(test_loader, model)
            print('test_acc: {:.3f}'.format(test_accuracy))
            accuracy[trial_i, fold_i] = test_accuracy

    print('The mean and std of accuracy at %d times %d folds: %f, %f' % (
    num_trial, num_fold, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))
    time_e = time.time()
    print('\nRunning time is', time_e - time_s, 'seconds.')
    print('Training is finished.')
