# import numpy as np
# import math
# import warnings
# import argparse
# import torch
# import torch.nn.functional as F
# from utils import *
# from models import *
# import faiss
# from faiss import normalize_L2
# import scipy
# import scipy.stats
# import time
# from sklearn.preprocessing import normalize
# from sklearn.neighbors import kneighbors_graph
# from tqdm import tqdm
# import os
# import json
#
# warnings.filterwarnings("ignore")
#
#
# def estimating_label_correlation_matrix(Y_P):
# 	num_class = Y_P.shape[1]
# 	n = Y_P.shape[0]
#
# 	R = np.zeros((num_class, num_class))
# 	for i in range(num_class):
# 		for j in range(num_class):
# 			if i == j:
# 				R[i][j] = 0
# 			else:
# 				if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0 :
# 					R[i][j] = 1e-5 # avoid divide zero error
# 				else:
# 					R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum())
# 	D_1_2 = np.diag(1. / np.sqrt(np.sum(R, axis=1)))
# 	L = D_1_2.dot(R).dot(D_1_2)
# 	L = np.nan_to_num(L)
#
# 	return L
#
#
# def build_graph_v2(X, k=10):
# 	if not args.no_verbose:
# 		print('Building Graph - V2 is not used')
# 	# kNN search for the graph
# 	W = kneighbors_graph(X, k, mode='distance', include_self=False, metric='euclidean')
#
# 	W.data = np.exp(- W.data ** 2 / (2 * 1))
#
# 	W = W + W.T
# 	W = W - scipy.sparse.diags(W.diagonal())
# 	S = W.sum(axis=1)
# 	S[S == 0] = 1
# 	D = np.array(1. / np.sqrt(S))
# 	D = scipy.sparse.diags(D.reshape(-1))
# 	Wn = D * W * D
#
# 	return Wn
#
# def build_graph(X, k=10, args=None):
# 	if not args.no_verbose:
# 		print('Building Graph - V1')
# 	# kNN search for the graph
# 	X = X.astype('float32')
# 	d = X.shape[1]
# 	#index = faiss.IndexFlatIP(d)  # build the index
# 	nlist = 100  # 簇的数量
# 	m = 8  # 子向量的数量（PQ 参数）
# 	nbits = 8  # 每个子向量的比特数（通常为 8）
#
# 	quantizer = faiss.IndexFlatIP(d)  # 使用内积作为距离度量
# 	index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
# 	# res = faiss.StandardGpuResources()
# 	# flat_config = faiss.GpuIndexFlatConfig()
# 	# flat_config.device = 0
# 	# index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index
#
# 	normalize_L2(X)
# 	index.add(X)
# 	N = X.shape[0]
# 	Nidx = index.ntotal
#
# 	c = time.time()
# 	D, I = index.search(X, k + 1)
# 	elapsed = time.time() - c
# 	if not args.no_verbose:
# 		print('kNN Search Time: ', elapsed)
#
# 	# Create the graph
# 	D = D[:, 1:] ** 3
# 	I = I[:, 1:]
# 	row_idx = np.arange(N)
# 	row_idx_rep = np.tile(row_idx, (k, 1)).T
# 	W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
# 	W = W + W.T
#
# 	# Normalize the graph
# 	W = W - scipy.sparse.diags(W.diagonal())
# 	S = W.sum(axis=1)
# 	S[S == 0] = 1
# 	D = np.array(1. / np.sqrt(S))
# 	D = scipy.sparse.diags(D.reshape(-1))
# 	Wn = D * W * D
#
# 	return Wn
#
# def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter):
#
# 	gamma = args.gamma  # learning rate
# 	Z = Y_P_train
#
# 	Z_g = torch.from_numpy(Z).float()
# 	Y_P_train_g = torch.from_numpy(Y_P_train).float()
# 	Y_pred_g = torch.from_numpy(Y_pred).float()
# 	L_g = torch.from_numpy(L).float()
#
# 	with torch.no_grad():
# 		for i in range(maxiter):
# 			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy()))
# 			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
# 			Z_g = Z_g - gamma * grad
#
# 	Z = Z_g.detach().cpu().numpy()
#
# 	min_max_scaler = preprocessing.MinMaxScaler()
# 	Z = min_max_scaler.fit_transform(Z)
#
# 	torch.empty_cache()
#
# 	return Z
#
#
# def run_model(args, data, model, optimizer, training_now=True, eval_every=5):
# 	res = None  # define the return value
# 	res_log = np.zeros(args.epochs * 150)
# 	overall_loss_log = np.zeros(args.epochs)
#
# 	X_train, Y_train, Y_P_train, X_test, Y_test = data
# 	batch_size = args.batch_size
# 	num_class = Y_P_train.shape[1]
# 	# def data and frequently used args
#
# 	# number of iteration times per epoch
# 	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
# 	# iter_per_epoch = 1 # without batch
#
# 	loss_pred = 0
#
# 	# counter
# 	iter_cnt = 0
#
# 	Y_pred_np = Y_P_train
# 	Y_lp_np = Y_P_train
#
# 	Wn = build_graph(X_train, k=args.neighbors_num, args=args)
# 	L = estimating_label_correlation_matrix(Y_P_train)
#
# 	best_score = 0
#
# 	for e in range(args.epochs):
# 		# shuffle indices
# 		train_indicies = np.arange(X_train.shape[0])
# 		np.random.shuffle(train_indicies)
#
# 		if args.using_lp:
# 			maxiter = args.maxiter
# 			Y_lp_np = label_propagation(args, Wn, L, Y_pred_np, Y_P_train, Y_lp_np
# 				, alpha=args.alpha, eta=args.eta, beta=args.beta, maxiter=maxiter)
#
# 		# keep track of losses
# 		model.train()
#
# 		for i in range(iter_per_epoch):
# 			optimizer.zero_grad()
#
# 			start_idx = (i * batch_size) % X_train.shape[0]
# 			idx = train_indicies[start_idx: start_idx + batch_size]
# 			# idx = train_indices # without batch
# 			# get minibatch indices
#
# 			# randomly select a mini-batch of data
# 			X = torch.from_numpy(X_train[idx, :]).float()
# 			Y_P = torch.from_numpy(Y_P_train[idx, :]).float()
# 			Y_lp = torch.from_numpy(Y_lp_np[idx, :]).float()
# 			X.requires_grad = False
# 			Y_P.requires_grad = False
# 			Y_lp.requires_grad = False
#
# 			Y_pred = model.forward(X)
#
# 			if args.using_lp:
# 				loss = F.mse_loss(Y_pred, Y_lp)
# 			else:
# 				loss = F.binary_cross_entropy(Y_pred, Y_P)
#
# 			loss.backward()
# 			optimizer.step()
# 			# approximating candidate labels
#
# 		Y_pred_np = test(args, X_train, num_class, model)
#
# 		Z = Y_lp_np
# 		model.eval()
#
# 		# testing
# 		if args.no_verbose:
# 			eval_every = 1
#
# 		if e % eval_every == 0:
# 			Y_pred = test(args, X_test, num_class, model)
# 			r_loss, h_loss, ap = evaluate(Y_test, Y_pred, threshold=args.threshold)
#
# 			if not args.no_verbose:
# 				print('Epoch %d: r_loss %.4f, h_loss %.4f, ap %.4f'
# 					% (e, r_loss, h_loss, ap))
#
# 	return res
#
#
# def test(args, X_test, num_class, model):
# 	X_test_tensor = torch.from_numpy(X_test).float()
# 	iter_per_epoch = int(math.ceil(X_test_tensor.shape[0] / args.batch_size))
# 	Y_pred = []
# 	with torch.no_grad():
# 		for i in range(iter_per_epoch):
# 			start_idx = (i * args.batch_size) % X_test_tensor.shape[0]
# 			X_batch = X_test_tensor[start_idx: start_idx + args.batch_size, :]
# 			Y_pred_batch = model.forward(X_batch)
# 			Y_pred += [Y_pred_batch.detach().cpu().numpy()]
#
# 	Y_pred = np.concatenate(Y_pred, axis=0)
# 	return Y_pred
#
#
# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# 	# Optimization options
# 	parser.add_argument('--epochs', default=300, type=int, metavar='N',
# 						help='number of total epochs to run')
# 	parser.add_argument('--batch-size', default=32, type=int, metavar='N',
# 						help='train batchsize')
# 	parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
# 						metavar='LR', help='initial learning rate')
# 	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
# 						help='SGD momentum (default: 0.9)')
# 	parser.add_argument('--using_lp', action='store_true')
# 	parser.add_argument('--using_prep', action='store_true')
# 	parser.add_argument('--class_cons', type=float, default=0.0)
# 	parser.add_argument('--weight_decay', type=float, default=5e-5)
# 	parser.add_argument('--neighbors_num', type=int, default=10)
# 	parser.add_argument('--threshold', type=float, default=0.7)
# 	parser.add_argument('--hidden_size', type=str, default='64,64')
# 	parser.add_argument('--gpuid', type=int, default=0)
#
# 	# main parameters
# 	parser.add_argument('--alpha', type=float, default=0.01)
# 	parser.add_argument('--eta', type=float, default=1)
# 	parser.add_argument('--beta', type=float, default=0.01)
#
# 	parser.add_argument('--maxiter', type=int, default=200)
# 	parser.add_argument('--gamma', type=float, default=0.01)
#
# 	parser.add_argument('--tr_rate', type=float, default=0.9)
#
# 	parser.add_argument('--no-verbose', action='store_true')
#
# 	parser.add_argument('--mode', type=int, default=0)
# 	parser.add_argument('--eval_every', type=int, default=5)
#
# 	args = parser.parse_args()
#
# 	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
#
# 	args.hidden_size = [int(n) for n in args.hidden_size.split(',')]
#
# 	target = 'music_emotion'
#
# 	print('Dataset: {}, lr: {}, using_lp: {}'.format(target, args.lr, args.using_lp))
#
# 	file_name = 'data/' + target
#
# 	if args.using_prep:
# 		print('Using preprocessed data ...')
# 		data = load_preprocessed_data(file_name)
# 	else:
# 		print('Using raw data ...')
# 		data = load_data(file_name, tr_rate=args.tr_rate)
# 	_, _, _, X_test, Y_test = data
# 	model = DeepNet(X_test.shape[1], Y_test.shape[1], args.hidden_size).cpu()
# 	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#
# 	print(' ')
# 	run_model(args, data, model, optimizer, eval_every=args.eval_every)
