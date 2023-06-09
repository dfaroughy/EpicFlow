
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def calculate_loss(model, context, data, args, loss_func=None, reduction=torch.mean):
	if not loss_func: loss_func = args.loss
	loss = reduction(loss_func(model, context, data, args))
	return loss

def epic_loss(model, context, batch, args):

	n = args.num_mc
	b = batch.size(0)
	#... repeat batch for MC integration

	batch = batch.repeat_interleave(n, dim=0)
	batch = torch.flatten(batch, end_dim=1)
	batch = batch.to(args.device)

	#... generate local and global contexts: 

	z_glob, z_loc, z_glob_in, z_loc_in = context.sample(n * b, args.num_const)  # on device already	
	z_glob = z_glob.repeat_interleave(args.num_const, dim=0)
	z_loc = torch.flatten(z_loc, end_dim=1)
	z_glob_in = z_glob_in.repeat_interleave(args.num_const, dim=0)
	z_loc_in = torch.flatten(z_loc_in, end_dim=1)
	z_context = torch.cat([z_glob, z_loc, z_glob_in, z_loc_in], 1)

	#... compute log_prob over batch with generated context:
	batch = batch.to(args.device)
	loss = model.log_prob(batch, context=z_context)   
	loss = torch.sum(loss.reshape((n * b, -1)), dim=1)
	loss = - torch.logsumexp(loss.reshape((-1, n)), dim=1)
	loss +=  torch.log(torch.tensor(1.0 if not n else n))

	return loss
