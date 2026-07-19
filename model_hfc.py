#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mix-loss:CrossEntropyLoss
from weight_loss import CrossEntropyLoss as CE

class GatedAttentionLayerV(nn.Module):
    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        # self.dim = dim
        self.dim = dim
        # self.linear = nn.Linear(dim, 1): 
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)
        return out_tanh

# GatedAttentionLayerU 
class GatedAttentionLayerU(nn.Module):
    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)
        return out_sigmoid

class MixLossAttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super( MixLossAttentionLayer, self).__init__() 
        self.dim = dim 
        self.linear = nn.Linear(dim, 1) 

    def forward(self, features, W_1, b_1, isMixLoss = True):
        if isMixLoss:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else: 
            context = features
            alpha = torch.zeros(features.size(0),1)
        return context, out_c, alpha

class MixAttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super( MixAttentionLayer, self).__init__() 
        self.dim = dim 
        self.linear = nn.Linear(dim, 1) 

    def forward(self, features, W_1, b_1, isMixLoss = True):
        if isMixLoss:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else: 
            context = features
            alpha = torch.zeros(features.size(0),1)
        return context, out_c, alpha
class MixGatedAttentionLayerV(nn.Module):
    def __init__(self, dim=512):
        super(MixGatedAttentionLayerV, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)
    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)
        return out_tanh,out

class MixGatedAttentionLayerU(nn.Module):
    def __init__(self, dim=512):
        super(MixGatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)
    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)
        return out_sigmoid,out




class HFCLossGatedAttention(nn.Module):
    def __init__(self, args):
        super(HFCLossGatedAttention, self).__init__()
        self.args = args
        self.L = self.args.L  
        self.D = 128
        self.K = 1
        self.nr_fea = self.args.nr_fea
        self.la = args.la
        self.mix_loss_rate = args.mix_loss_rate
        self.fc = nn.Conv2d(in_channels=50, out_channels=self.args.nr_class, kernel_size=1, stride=1, bias=False)
        # 1st extractor
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # second extractor
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )
        # third extractor
        self.feature_extractor_part1_small = nn.Sequential(
            nn.Linear(self.nr_fea, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.att_layer = MixLossAttentionLayer(self.L)
        self.linear = nn.Linear(self.L*self.K, self.args.nr_class)
        self.criterion = torch.nn.CrossEntropyLoss(size_average=True)
        self.weight_criterion = CE(aggregate='mean')
        
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_V2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        self.linear_U2 = nn.Linear(self.args.nr_class, self.args.nr_class)
        
        self.att_layer_MV = MixGatedAttentionLayerV(self.L)
        self.att_layer_MU = MixGatedAttentionLayerU(self.L)
        self.linear_MV = nn.Linear(self.L * self.K, self.args.nr_class)

        self.attention_weights = nn.Sequential(
            nn.Linear(self.args.nr_class, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.args.nr_class),
            nn.Sigmoid()
        )
        # 论文ψ2 包特征维度对齐映射网络
        self.psi2 = nn.Linear(self.L, self.L)
        # 混合特征分类头，输入维度 2*L (拼接后的f)
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.L, self.args.nr_class),
            nn.Sigmoid()
        )
    #     Epoch-adaptive noisy instance filtering 动态实例筛选
    def epoch_adaptive_noisy_instance_filtering(
        self, 
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
        
        return selected.copy()
    
    
    def forward(self, X, args,isMixLoss = True):
        Y_prob = None
        A = None
        out_c = None 
        if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
            X = X.squeeze(0)
            H = self.feature_extractor_part1(X)
            H2 = H.view(-1, 50 * 4 * 4)
            H2 = self.feature_extractor_part2(H2)
        else: 
            X = X.float()
            X = X.view(-1, args.nr_fea)
            H = self.feature_extractor_part1_small(X)
            H2 = H
        # ========= 1 门控注意力计算A =========
        if not isMixLoss:
            A_V = self.att_layer_V(H2, self.linear_V.weight, self.linear_V.bias)
            A_U = self.att_layer_U(H2, self.linear_V.weight, self.linear_V.bias)
            A = self.attention_weights(A_V * A_U) 
        if isMixLoss:
            A_MV,MidV_out = self.att_layer_MV(H2, self.linear_MV.weight, self.linear_MV.bias) 
            A_MU,MidU_out = self.att_layer_MU(H2, self.linear_MV.weight, self.linear_MV.bias)
            A_MUV=A_MV * A_MU
            out_c = A_MUV
            A = self.attention_weights(A_MUV) 
        A = torch.transpose(A, 1, 0) 
        A = torch.sigmoid(A)
        # ========= 2 原全局Bag特征 z =========
        z = torch.mm(A, H2) / torch.sum(A)
        M = z   
        Y_prob = self.classifier(M)
        # ========= 3 新增：动态过滤+实例均值 h_bar 【论文公式5】 =========
        # 调用过滤函数获取保留实例下标
        keep_idx_np = self.epoch_adaptive_noisy_instance_filtering(
            attention_scores=A.detach().cpu().numpy(),
            epoch=args.current_epoch,
            update_interval=10,
            tau0=0.2,
            delta=0.02,
            tau_max=0.6,
            min_instances=1,
            max_instances=H2.size(0)
        )
        # 拷贝数组，消除负stride
        keep_idx = keep_idx_np.copy()
        keep_idx = torch.tensor(keep_idx, device=device)
        h_keep = H2[keep_idx]  # 筛选后的干净实例特征
        h_bar = torch.mean(h_keep, dim=0, keepdim=True)  # 纯化实例均值 h̄
        # ========= 4 包特征映射 ψ2 得到 z_tilde 【论文公式4】 =========
        z_tilde = self.psi2(z)
        # ========= 5 混合特征拼接 f = [z_tilde; h_bar] 【论文公式6】 =========
        f = torch.cat([z_tilde, h_bar], dim=1)
        f = f.squeeze() 
        # ========= 6 混合特征分类输出 =========
        instance_Y_prob = self.instance_classifier(h_bar)
        out_c =self.instance_classifier(H2)
        #  out_c 实例输出损失        
        return Y_prob, A ,out_c, instance_Y_prob,f,keep_idx
       
    def compute_focusing_loss(self,attentions, pred_logits, S_mask, T=1.5, beta=2.0, eps=1e-8):
        """
        3.4.1 Focusing Regularization L_fo = L_sp-feat + β*L_sp-label
        Args:
            attentions: 各包实例注意力权重 a_ij, shape [batch, num_ins]
            pred_logits: 模型预测原始logits [batch, num_class]
            S_mask: 候选标签掩码 S_i (1=候选标签,0=非候选) [batch, num_class]
            T: 聚焦因子 T>1 论文公式超参
            beta: 平衡特征/标签正则权重 β
            eps: 数值稳定项，防止log(0)
        Returns:
            L_fo: 总聚焦正则损失
            L_sp_feat: 注意力权重聚焦损失
            L_sp_label: 标签概率L1聚焦损失
        """
        batch_size = attentions.shape[0]
        # ========== 1. 计算注意力聚焦损失 L_sp-feat 公式(9) ==========
        a_pow = torch.pow(attentions, T)  # a_ij^T
        log_a = torch.log(a_pow + eps)   # 防止log(0)
        feat_loss_per_sample = torch.sum(a_pow * log_a, dim=1)
        L_sp_feat = torch.sum(feat_loss_per_sample) / batch_size

        # ========== 2. 计算标签聚焦损失 L_sp-label 公式(10) ==========
        prob = torch.softmax(pred_logits, dim=-1)  # 得到预测概率P_i
        prob_masked = prob * S_mask                # P_i ⊙ S_i 逐元素乘掩码
        l1_per_sample = torch.norm(prob_masked, p=1, dim=-1)
        L_sp_label = torch.sum(l1_per_sample) / batch_size

        # ========== 3. 总聚焦损失 L_fo 公式(11) ==========
        L_fo = L_sp_feat + beta * L_sp_label
        return L_fo, L_sp_feat, L_sp_label
    
    


    def evaluate_objective(self, X, args,isMixLoss = False):
        '''
        model testing
        '''
        if args.isMixLoss:
            isMixLoss = True
        Y_prob, _, _, _, _,_  = self.forward(X, args,isMixLoss)
        Y_prob = F.softmax(Y_prob, dim=1)
        return Y_prob
    
    def full_hfc_loss_deepm(self, args,A,out_c, prediction,keep_idx, target, X_recon_iter,aggre_data_iter,Y_P_train_iter,Y_recon_iter, isMixLoss = False):

        Y_candiate = torch.zeros(target.shape).to(device)
        Y_candiate[target > 0] = 1
        prediction_mask = prediction * Y_candiate
        new_prediction = prediction_mask / prediction_mask.sum(dim=1).repeat(prediction_mask.size(1), 1).transpose(0, 1)
        entropy_Y = - target * torch.log(prediction)
        entropy_A = - A * torch.log(A)
        entropy_X_sum = 0 
        instance_loss = 0
        if isMixLoss and args.current_epoch > (args.epochs/2):
            mask = torch.ones_like(A)
            mask[0, keep_idx] = 0.0
            A_selected = A * mask
            target_max_value = torch.argmax(target).to(device) 
            instance_labels = target_max_value * torch.squeeze(torch.ones(A.size(1),1)).type(torch.LongTensor).to(device) 
            instance_loss = self.weight_criterion(out_c, instance_labels, weights=torch.squeeze(A_selected))

        manifold_loss = 0
        if Y_recon_iter is not None:
            Y_recon_iter = torch.from_numpy(Y_recon_iter).float().detach().cuda()
            Y_train_pred_iter = new_prediction.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]
            manifold_loss = F.mse_loss(Y_train_pred_iter, Y_recon_iter)
        
        fo_loss = 0   
        if args.current_epoch > (args.epochs/2):          
            idx_candidate = torch.squeeze(torch.nonzero(torch.squeeze(Y_candiate)))
            prob_candidate = torch.squeeze(prediction_mask)[idx_candidate]
            fo_loss = torch.norm(prob_candidate, p=1, dim=0)
        # 超参可从args自定义，这里默认T=1.5, beta=2.0
            fo_loss, _, _ = self.compute_focusing_loss(
                attentions=A,
                pred_logits=prediction,
                S_mask=Y_candiate,
                T=getattr(args, "T_focus", 1.5),
                beta=getattr(args, "beta_focus", 2.0)
            )
        # total loss
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)+ args.mix_loss_rate * instance_loss+args.lambda_recon_rate * manifold_loss + args.mu * fo_loss 
        return new_prediction, loss

    def calculate_objective_hfc_deepm(self,args,X, Y,
                                       X_recon_iter,aggre_data_iter,Y_P_train_iter,Y_recon_iter,
                                       isMixLoss = False):
        '''
        calculate the full loss, weighted partial labels, and attention scores
        '''
        if args.isMixLoss:
            isMixLoss = True
        
        Y = Y.reshape(-1)
        Y_prob, A , out_c, instance_Y_prob,f,keep_idx = self.forward(X, args,isMixLoss)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        Y_prob = F.softmax(Y_prob, dim=1)
        new_prob, loss = self.full_hfc_loss_deepm(args,A, out_c, Y_prob,keep_idx, Y, X_recon_iter,aggre_data_iter,Y_P_train_iter,Y_recon_iter,isMixLoss)
        return loss, new_prob, A,f
    