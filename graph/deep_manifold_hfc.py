import numpy as np
import math
import warnings
import argparse
import torch
import torch.nn.functional as F
import os  
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
grandparent_path = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, grandparent_path)
sys.path.insert(1, parent_path)
sys.path.insert(2, current_path)
print('first dir:', sys.path[0])


from utils import *
from deep_manifold_models import *
import faiss
from faiss import normalize_L2
import scipy
import scipy.stats
import time
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import os
import json
warnings.filterwarnings("ignore")

def estimating_label_correlation_matrix(Y_P):
	num_class = Y_P.shape[1]  
	n = Y_P.shape[0]  

	R = np.zeros((num_class, num_class)) 
	for i in range(num_class):
		for j in range(num_class):
			if i == j:
				R[i][j] = 0
			else:
				if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0 :
					R[i][j] = 1e-5 
				else:
					R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum()) 
	D_1_2 = np.diag(1. / np.sqrt(np.sum(R, axis=1)))  
	L = D_1_2.dot(R).dot(D_1_2) 

	L = np.nan_to_num(L)
	return L



def build_graph_v2(X, k=10):
	if not args.no_verbose:
		print('Building Graph - V2 is not used')
	W = kneighbors_graph(X, k, mode='distance', include_self=False, metric='euclidean')
	W.data = np.exp(- W.data ** 2 / (2 * 1))


	W = W + W.T
	W = W - scipy.sparse.diags(W.diagonal())
	S = W.sum(axis=1)
	S[S == 0] = 1
	D = np.array(1. / np.sqrt(S))
	D = scipy.sparse.diags(D.reshape(-1))
	Wn = D * W * D

	return Wn

def build_graph(X, k=10, args=None):
	if not args.no_verbose:
		print('Building Graph - V1')
	# kNN search for the graph
	X = X.astype('float32')
	d = X.shape[1]
	# init FAISS GPU index
	res = faiss.StandardGpuResources()
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

	print(f"Before normalization - X stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
	print(f"X contains NaN: {np.isnan(X).any()}")
	print(f"X contains Inf: {np.isinf(X).any()}")
	print(f"X contains zeros: {(X == 0).all(axis=1).any()}")
	
	normalize_L2(X)
	
	# 检查归一化后的数据
	print(f"After normalization - X stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
	print(f"X contains NaN after norm: {np.isnan(X).any()}")
	print(f"X contains Inf after norm: {np.isinf(X).any()}")
	print(f"X L2 norms: min={np.linalg.norm(X, axis=1).min():.6f}, max={np.linalg.norm(X, axis=1).max():.6f}")
	index.add(X)
	N = X.shape[0]
	Nidx = index.ntotal
	c = time.time()
	D, I = index.search(X, k + 1) 
	elapsed = time.time() - c
	if not args.no_verbose:
		print('kNN Search Time: ', elapsed)
	
	# 调试信息
	print(f"X shape: {X.shape}, k: {k}, N: {N}")
	print(f"D shape: {D.shape}, I shape: {I.shape}")
	print(f"D range: [{D.min():.4f}, {D.max():.4f}]")
	print(f"I range: [{I.min()}, {I.max()}]")
	print(f"First few I values: {I[:3, :]}")
	D = D[:, 1:] ** 3
	I = I[:, 1:]

	row_idx = np.arange(N)
	row_idx_rep = np.tile(row_idx, (k, 1)).T
	W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
	W = W + W.T

	W = W - scipy.sparse.diags(W.diagonal())
	S = W.sum(axis=1)
	S[S == 0] = 1
	D = np.array(1. / np.sqrt(S))
	D = scipy.sparse.diags(D.reshape(-1))
	Wn = D * W * D

	return Wn
def label_propagation_with_x_reconstruction(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter,X_train =None):
	gamma = args.gamma  # learning rate

	Z = Y_P_train

	Z_g = torch.from_numpy(Z).float().detach().cuda()
	Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
	Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
	L_g = torch.from_numpy(L).float().detach().cuda()
	with torch.no_grad():
		for i in range(maxiter):
			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
			Z_g = Z_g - gamma * grad
		if X_train is not None:
			X_recon_new = torch.from_numpy(Wn.dot(torch.from_numpy(X_train))).detach().cuda() 

	Z_g = torch.softmax(Z_g, dim=1)

	Z = Z_g.detach().cpu().numpy()

	torch.cuda.empty_cache()
	return Z,X_recon_new

def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter):

	gamma = args.gamma  # learning rate
	Z = Y_P_train

	Z_g = torch.from_numpy(Z).float().detach().cuda()
	Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
	Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
	L_g = torch.from_numpy(L).float().detach().cuda()

	with torch.no_grad():
		for i in range(maxiter):
			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
			Z_g = Z_g - gamma * grad

	Z_g = torch.softmax(Z_g, dim=1)
	Z = Z_g.detach().cpu().numpy()
	torch.cuda.empty_cache()
	return Z

def run_model(args, data, model, optimizer, training_now=True, eval_every=5):
	res = None  
	res_log = np.zeros(args.epochs * 150)
	overall_loss_log = np.zeros(args.epochs)
	X_train, Y_train, Y_P_train, X_test, Y_test = data
	batch_size = args.batch_size
	num_class = Y_P_train.shape[1]
	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
	loss_pred = 0
	iter_cnt = 0
	Y_pred_np = Y_P_train
	Y_lp_np = Y_P_train
	Wn = build_graph(X_train, k=args.neighbors_num, args=args)
	L = estimating_label_correlation_matrix(Y_P_train)

	best_score = 0

	for e in range(args.epochs):
		train_indicies = np.arange(X_train.shape[0])
		np.random.shuffle(train_indicies)

		if args.using_lp:
			maxiter = args.maxiter
			Y_lp_np = label_propagation(args, Wn, L, Y_pred_np, Y_P_train, Y_lp_np
				, alpha=args.alpha, eta=args.eta, beta=args.beta, maxiter=maxiter) 
		model.train()

		for i in range(iter_per_epoch):
			optimizer.zero_grad()
			start_idx = (i * batch_size) % X_train.shape[0]
			idx = train_indicies[start_idx: start_idx + batch_size]
			X = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
			Y_P = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
			Y_lp = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
			X.requires_grad = False
			Y_P.requires_grad = False
			Y_lp.requires_grad = False
			Y_pred = model.forward(X)
			if args.using_lp:
				loss = F.mse_loss(Y_pred, Y_lp)
			else:
				loss = F.binary_cross_entropy(Y_pred, Y_P)
			loss.backward()
			optimizer.step()

		Y_pred_np = test(args, X_train, num_class, model)

		Z = Y_lp_np
		model.eval()

		if args.no_verbose:
			eval_every = 1

		if e % eval_every == 0:
			Y_pred = test(args, X_test, num_class, model)
			r_loss, h_loss, ap = evaluate(Y_test, Y_pred, threshold=args.threshold)

			if not args.no_verbose:
				print('Epoch %d: r_loss %.4f, h_loss %.4f, ap %.4f' 
					% (e, r_loss, h_loss, ap))
	return res

def test(args, X_test, num_class, model):
	X_test_tensor = torch.from_numpy(X_test).float().cuda().detach()
	iter_per_epoch = int(math.ceil(X_test_tensor.shape[0] / args.batch_size))
	Y_pred = []
	with torch.no_grad():
		for i in range(iter_per_epoch):
			start_idx = (i * args.batch_size) % X_test_tensor.shape[0]
			X_batch = X_test_tensor[start_idx: start_idx + args.batch_size, :]
			Y_pred_batch = model.forward(X_batch)
			Y_pred += [Y_pred_batch.detach().cpu().numpy()]
	Y_pred = np.concatenate(Y_pred, axis=0) 
	return Y_pred


def deepm_train_global_prepare(args, X_train, Y_P_train):

	if np.isnan(X_train).any():
		print("warning data contain NaN value")
	Wn = build_graph(X_train, k=args.neighbors_num, args=args)
	L = estimating_label_correlation_matrix(Y_P_train)  

	if args.using_lp:
		maxiter = args.maxiter
		Y_recon,X_recon = label_propagation_with_x_reconstruction(args, Wn, L, Y_P_train, Y_P_train, None
									, alpha=args.alpha, eta=args.eta, beta=args.beta,
									maxiter=maxiter,X_train=X_train)  
	return Y_P_train, Y_recon, Wn, L,X_recon

def deepm_train_iter(args, model_deepm, X_train_iter, Y_train_P_iter, Y_train_lp_iter):
	Y_train_P_iter = torch.from_numpy(Y_train_P_iter).float().detach().cuda()
	Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
	X_train_iter.requires_grad = False
	Y_train_P_iter.requires_grad = False
	Y_train_lp_iter.requires_grad = False
	Y_pred_iter,A = model_deepm.forward(X_train_iter,args)
	Y_pred_iter = Y_pred_iter.squeeze(0)  
	if args.using_lp:
		deepm_loss = F.mse_loss(Y_pred_iter, Y_train_lp_iter)
	else:
		deepm_loss = F.binary_cross_entropy(Y_pred_iter, Y_train_P_iter)
	return deepm_loss

def deepm_run(args, data, model_deepm, optimizer, training_now=True, eval_every=5):
	res = None  
	res_log = np.zeros(args.epochs * 150)
	overall_loss_log = np.zeros(args.epochs)
	X_train, Y_train, Y_P_train, X_test, Y_test = data
	batch_size = args.batch_size
	num_class = Y_P_train.shape[1]
	loss_pred = 0
	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
	iter_cnt = 0
	Y_pred_np, Y_lp_np, Wn, L = deepm_train_global_prepare(args, X_train, Y_P_train)
	best_score = 0
	train_indicies = np.arange(X_train.shape[0])
	np.random.shuffle(train_indicies)
	model_deepm.train()
	for i in range(iter_per_epoch):
		optimizer.zero_grad()
		start_idx = (i * batch_size) % X_train.shape[0]
		idx = train_indicies[start_idx: start_idx + batch_size]
		X_train_iter = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
		Y_train_P_iter = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
		Y_train_lp_iter = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
		X_train_iter.requires_grad = False
		Y_train_P_iter.requires_grad = False
		Y_train_lp_iter.requires_grad = False
		deepm_loss = deepm_train_iter(args, model_deepm, X_train_iter, Y_train_P_iter, Y_train_lp_iter)
		deepm_loss.backward()
		optimizer.step()
	Y_pred_np = test(args, X_train, num_class, model_deepm)

	Z = Y_lp_np
	model_deepm.eval()

	if args.no_verbose:
		eval_every = 1

	Y_test_pred = test(args, X_test, num_class, model_deepm)
	r_loss, h_loss, ap = evaluate(Y_test, Y_test_pred, threshold=args.threshold)

	if not args.no_verbose:
		print(' r_loss %.4f, h_loss %.4f, ap %.4f'
			  % (r_loss, h_loss, ap))
	return Y_pred_np, Y_test_pred, r_loss, h_loss, ap





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
	parser.add_argument('--epochs', default=300, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--batch-size', default=32, type=int, metavar='N',
						help='train batchsize')
	parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--using_lp', action='store_true')
	parser.add_argument('--using_prep', action='store_true')
	parser.add_argument('--class_cons', type=float, default=0.0)
	parser.add_argument('--weight_decay', type=float, default=5e-5)
	parser.add_argument('--neighbors_num', type=int, default=10)
	parser.add_argument('--threshold', type=float, default=0.7)
	parser.add_argument('--hidden_size', type=str, default='64,64')
	parser.add_argument('--gpuid', type=int, default=0)
	parser.add_argument('--alpha', type=float, default=0.01)
	parser.add_argument('--eta', type=float, default=1)
	parser.add_argument('--beta', type=float, default=0.01)
	parser.add_argument('--maxiter', type=int, default=200)
	parser.add_argument('--gamma', type=float, default=0.01)
	parser.add_argument('--tr_rate', type=float, default=0.9)
	parser.add_argument('--no-verbose', action='store_true')
	parser.add_argument('--mode', type=int, default=0)
	parser.add_argument('--eval_every', type=int, default=5)
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
	args.hidden_size = [int(n) for n in args.hidden_size.split(',')]
	target = 'music_emotion'
	print('Dataset: {}, lr: {}, using_lp: {}'.format(target, args.lr, args.using_lp))
	file_name = 'yourpath'+'data/' + target
	if args.using_prep:
		print('Using preprocessed data ...')
		data = load_preprocessed_data(file_name)
	else:
		print('Using raw data ...')
		data = load_data(file_name, tr_rate=args.tr_rate)
	_, _, _, X_test, Y_test = data
	model = DeepNet(X_test.shape[1], Y_test.shape[1], args.hidden_size).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	print(' ')
	deepm_run(args, data, model, optimizer, eval_every=args.eval_every)