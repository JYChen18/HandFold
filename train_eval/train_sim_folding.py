'''
training MSRA
'''
import argparse
import os
import random
import time
import logging
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset_sim import SimDataset
from network_msra_folding import PointNet_Plus
from utils import group_points, rotate_point_cloud_by_angle_flip

if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()
	parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
	parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
	parser.add_argument('--nepoch', type=int, default=80, help='number of epochs to train for')
	parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
	parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
	parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (SGD only)')
	parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

	parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
	parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
	parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
	parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
	parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
	parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
	parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
	parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
	parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
	parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

	parser.add_argument('--save_root_dir', type=str, default='/data/h2o_data/runs/HandFlod/5.8_Sim',  help='output folder')
	parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
	parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')

	opt = parser.parse_args()
	print (opt)

	torch.cuda.set_device(opt.main_gpu)

	opt.manualSeed = 1
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)

	save_dir = os.path.join(opt.save_root_dir, 'simgrasp')


	def _debug(model):
		module = model.netR_1
		print(module.named_paramters())
	try:
		os.makedirs(save_dir)
	except OSError:
		pass

	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
						filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
	logging.info('======================================================')

	# 1. Load data
	train_data = SimDataset(train = True)
	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
											shuffle=True, num_workers=int(opt.workers), pin_memory=False)
											
	test_data = SimDataset(train = False)
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
											shuffle=False, num_workers=int(opt.workers), pin_memory=False)
											
	print('#Train data:', len(train_data), '#Test data:', len(test_data))
	# print (opt)

	# 2. Define model, loss and optimizer
	netR = PointNet_Plus(opt)

	if opt.ngpu > 1:
		netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
		netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
		netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))
	if opt.model != '':
		# netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
		saved_model = torch.load(os.path.join(save_dir, opt.model))
		model_keys = list(saved_model.keys())
		ckpt = OrderedDict()
		for key in model_keys:
			ckpt[key.replace('.module.', '.')] = saved_model[key]
		netR.load_state_dict(ckpt)
	netR.cuda()
	# print(netR)

	weight_decay_list = (param for name, param in netR.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
	no_decay_list = (param for name, param in netR.named_parameters() if name[-4:] == 'bias' or "bn" in name)
	parameters = [{'params': weight_decay_list},
				{'params': no_decay_list, 'weight_decay': 0.}]


	# criterion = nn.MSELoss(size_average=True).cuda()
	def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
		beta = 1. / (sigma ** 2)
		diff = torch.abs(input - target)
		cond = diff < beta
		loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
		if reduce:
			return torch.sum(loss) / normalizer
		return torch.sum(loss, dim=1) / normalizer
	criterion = smooth_l1_loss

	optimizer = optim.Adam(parameters, lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06, weight_decay=opt.weight_decay)
	if opt.optimizer != '':
		optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
	scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

	# 3. Training and testing
	for epoch in range(opt.nepoch):
		scheduler.step(epoch)
		print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
		# 3.1 switch to train mode
		torch.cuda.synchronize()
		netR.train()
		train_mse = 0.0
		train_mse_wld = 0.0
		timer = time.time()

		for i, data in enumerate(tqdm(train_dataloader, 0)):
			if len(data[0]) == 1:
				continue
			torch.cuda.synchronize()       
			# 3.1.1 load inputs and targets
			points, volume_length, gt_xyz = data
			points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
			volume_length = volume_length.unsqueeze(-1)

			permutation = torch.randperm(points.size(1))
			points = points[:,permutation,:]

			points, gt_xyz = rotate_point_cloud_by_angle_flip(points, gt_xyz.view(-1, 21, 3), False)
			gt_xyz = gt_xyz.view(-1, 63)

			# print(gt_xyz.size())
			# points: B * 1024 * 6; target: B * 42
			inputs_level1, inputs_level1_center = group_points(points, opt)
			inputs_level1, inputs_level1_center = Variable(inputs_level1, requires_grad=False), Variable(inputs_level1_center, requires_grad=False)

			# 3.1.2 compute output
			optimizer.zero_grad()
			# estimation = netR(inputs_level1, inputs_level1_center)
			# loss = criterion(estimation, gt_xyz)*63
			fold1, fold2, estimation = netR(inputs_level1, inputs_level1_center)
			loss = (criterion(estimation, gt_xyz)*2+criterion(fold1, gt_xyz)+criterion(fold2, gt_xyz))*63

			# 3.1.3 compute gradient and do SGD step
			loss.backward()
			optimizer.step()
			torch.cuda.synchronize()
			
			# 3.1.4 update training error
			train_mse = train_mse + loss.item()*len(points)
			
			# 3.1.5 compute error in world cs      
			outputs_xyz = estimation
			diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
			diff_sum = torch.sum(diff,2)
			diff_sum_sqrt = torch.sqrt(diff_sum)
			diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
			diff_mean_wld = torch.mul(diff_mean,volume_length)
			train_mse_wld = train_mse_wld + diff_mean_wld.sum()

		# time taken
		torch.cuda.synchronize()
		timer = time.time() - timer
		timer = timer / len(train_data)
		print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

		# print mse
		train_mse = train_mse / len(train_data)
		train_mse_wld = train_mse_wld / len(train_data)
		print('mean-square error of 1 sample: %f, #train_data = %d' %(train_mse, len(train_data)))
		print('average estimation error in world coordinate system: %f (m)' %(train_mse_wld))

		torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
		torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
		
		# 3.2 switch to evaluate mode
		torch.cuda.synchronize()
		netR.eval()
		test_mse = 0.0
		test_wld_err = 0.0
		timer = time.time()
		for i, data in enumerate(tqdm(test_dataloader, 0)):
			torch.cuda.synchronize()
			with torch.no_grad():
				# 3.2.1 load inputs and targets
				points, volume_length, gt_xyz = data
				points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
				gt_xyz = gt_xyz.view(-1, 63)
				volume_length = volume_length.unsqueeze(-1)
				# points: B * 1024 * 6; target: B * 42
				inputs_level1, inputs_level1_center = group_points(points, opt)
				inputs_level1, inputs_level1_center = Variable(inputs_level1, volatile=True), Variable(inputs_level1_center, volatile=True)
			
				# 3.2.2 compute output
				fold1, fold2, estimation = netR(inputs_level1, inputs_level1_center)
				loss = (criterion(estimation, gt_xyz)+criterion(fold1, gt_xyz)+criterion(fold2, gt_xyz))*63

			torch.cuda.synchronize()
			test_mse = test_mse + loss.item()*len(points)

			# 3.2.3 compute error in world cs        
			# outputs_xyz = test_data.PCA_mean.expand(estimation.data.size(0), test_data.PCA_mean.size(1))
			outputs_xyz = estimation
			diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
			diff_sum = torch.sum(diff,2)
			diff_sum_sqrt = torch.sqrt(diff_sum)

			diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
			diff_mean_wld = torch.mul(diff_mean,volume_length)
			test_wld_err = test_wld_err + diff_mean_wld.sum()
		# time taken
		torch.cuda.synchronize()
		timer = time.time() - timer
		timer = timer / len(test_data)
		print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
		# print mse
		test_mse = test_mse / len(test_data)
		print('mean-square error of 1 sample: %f, #test_data = %d' %(test_mse, len(test_data)))
		test_wld_err = test_wld_err / len(test_data)
		print('average estimation error in world coordinate system: %f (m)' %(test_wld_err))
		# log
		logging.info('Epoch#%d: train error=%e, train wld error = %f m, test error=%e, test wld error = %f m, lr = %f' %(epoch, train_mse, train_mse_wld, test_mse, test_wld_err, scheduler.get_lr()[0]))

