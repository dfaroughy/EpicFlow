import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import copy
from copy import deepcopy
import argparse
import json
import h5py
# from jetnet.datasets import JetNet


from EpicFlow.utils.base import logit, expit, Id,  make_dir, copy_parser, save_arguments
from EpicFlow.models.flows.norm_flows import masked_autoregressive_flow, coupling_flow
from EpicFlow.models.context.epic import Epic
from EpicFlow.models.training import Train_Model, sampler
from EpicFlow.models.loss import epic_loss
from EpicFlow.data.transform import get_jet_data

sys.path.append("../")
torch.set_default_dtype(torch.float64)

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the deconvolution model')

params.add_argument('--workdir',      help='working directory', type=str)
params.add_argument('--device',       default='cuda:0',          help='where to train')
params.add_argument('--dim',          default=3,                 help='dimensionalaty of data: (pT, eta, phi)', type=int)
params.add_argument('--num_mc',       default=100,               help='number of MC samples for integration', type=int)
params.add_argument('--loss',         default=epic_loss,         help='loss function')

#...flow params:

params.add_argument('--flow',         default='MAF',        help='type of flow model: coupling or MAF', type=str)
params.add_argument('--dim_flow',     default=3,            help='dimensionalaty of input features for flow, usually same as --dim', type=int)
params.add_argument('--flow_func',    default='RQSpline',   help='type of flow transformation: affine or RQSpline', type=str)
params.add_argument('--coupl_mask',   default='checkers',   help='mask type [only for coupling flows]: mid-split or checkers', type=str)
params.add_argument('--permutation',  default='inverse',    help='type of fixed permutation between flows: n-cycle or inverse', type=str)
params.add_argument('--num_flows',    default=5,            help='num of flow layers', type=int)
params.add_argument('--dim_hidden',   default=128,          help='dimension of hidden layers', type=int)
params.add_argument('--num_spline',   default=20,           help='num of spline for rational_quadratic', type=int)
params.add_argument('--num_blocks',   default=2,            help='num of MADE blocks in flow', type=int)

#...training params:

params.add_argument('--batch_size',   default=64,             help='size of training/testing batch', type=int)
params.add_argument('--batch_steps',  default=0,              help='set the number of sub-batch steps for gradient accumulation', type=int)
params.add_argument('--test_size',    default=0.2,            help='fraction of testing data', type=float)
params.add_argument('--max_epochs',   default=3,              help='max num of training epochs', type=int)
params.add_argument('--max_patience', default=20,             help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',           default=1e-4,           help='learning rate of generator optimizer', type=float)
params.add_argument('--activation',   default=F.leaky_relu,   help='activation function for neuarl networks')
params.add_argument('--batch_norm',   default=True,           help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',      default=0.1,            help='dropout probability', type=float)

#... data params:

params.add_argument('--jet',          default='top-quark', help='jet data type: top or quark', type=str)
params.add_argument('--num_jets',     default=100000,      help='number of max constituents per jet', type=int)
params.add_argument('--num_const',    default=30,          help='number of max constituents per jet', type=int)
params.add_argument('--num_gen',      default=10000,       help='number of sampled jets from model', type=int)
params.add_argument('--num_hardest',  default=None,        help='keep the hardest contituents in each jets', type=int)

#... context params:

params.add_argument('--context',          default='epic',       help='type of context model: deepset, epic', type=str)
params.add_argument('--dim_z',            default=None,         help='dim of total context feature', type=int)
params.add_argument('--dim_loc_z',        default=3,            help='dim of local context feature', type=int)
params.add_argument('--dim_glob_z',       default=10,           help='dim of global context feature', type=int)
params.add_argument('--dim_hidden_z',     default=128,          help='dimension of hidden context layers', type=int)
params.add_argument('--pooling',          default='sum_mean',   help='pooling type: sum, sum_mean, attention', type=str)
params.add_argument('--num_epic_layers',  default=4,            help='number of epic layers', type=int)


####################################################################################################################

if __name__ == '__main__':

    #...create working folders and save args

    args = params.parse_args()

    if args.jet == 'top-quark' :  data_file =  "./data/t.hdf5"
    if args.jet == 'quark' :  data_file =  "./data/q.hdf5"

    args.workdir = make_dir('Results_Epic_Flow', sub_dirs=['data_plots', 'result_plots'], overwrite=False)
    save_arguments(args, name='inputs.json')
    print("#================================================")
    print("INFO: working directory: {}".format(args.workdir))
    print("#================================================")

    #...get datasets

    JetNet = h5py.File(data_file)
    jets = get_jet_data(JetNet['particle_features'], transform_pt=logit, remove_mask=True)
    jets = np.array(jets[args.num_const][:args.num_jets], dtype=np.float32)[:,:args.num_hardest]

    #...define models

    if args.flow == 'MAF': flow = masked_autoregressive_flow(args) 
    elif args.flow == 'coupling':  flow = coupling_flow(args) 

    if args.context == 'epic': 
        context = Epic(args)
        args.dim_z = 2*(args.dim_glob_z + args.dim_loc_z)

    flow = flow.to(args.device)
    context = context.to(args.device)

    #...store parameters

    save_arguments(args, name='inputs.json')   

    #...Prepare train/test samples

    train, test = train_test_split(jets, test_size=args.test_size,  random_state=9999)
    train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    test_sample = DataLoader(dataset=torch.Tensor(test), batch_size=args.batch_size, shuffle=False)

    # #...pretrain the flow to fit the noisy data before deconvolution

    Train_Model(flow, context, train_sample, test_sample, args)

    # sample = sampler(flow, num_samples=args.num_gen)

