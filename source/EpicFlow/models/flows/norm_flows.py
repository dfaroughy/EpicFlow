import torch
import numpy as np
import nflows
from nflows import flows, transforms, distributions
from EpicFlow.models.flows.permutation import permutation_layer

def masked_autoregressive_flow(args,
                               RQS_tail_bound=12, 
                               use_residual_blocks=True, 
                               random_mask=False, 
                               save_architecture=True):
   
    dim = args.dim_flow
    flow_model = args.flow + '_' + args.flow_func
    list_transforms = []

    print('INFO: model -> {}, {} transform'.format(args.flow, args.flow_func))

    for _ in range(args.num_flows):   
        if flow_model=='MAF_affine':
            flow = nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                                                features=dim,
                                                hidden_features=args.dim_hidden,
                                                context_features=args.dim_z,
                                                num_blocks=args.num_blocks,
                                                use_residual_blocks=use_residual_blocks,
                                                random_mask=random_mask,
                                                activation=args.activation,
                                                dropout_probability=args.dropout,
                                                use_batch_norm=args.batch_norm
                                                )
        elif flow_model=='MAF_RQSpline':  
            flow = nflows.transforms.autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                                                features=dim,
                                                hidden_features=args.dim_hidden,
                                                context_features=args.dim_z,
                                                num_blocks=args.num_blocks,
                                                use_residual_blocks=use_residual_blocks,
                                                random_mask=random_mask,
                                                activation=args.activation,
                                                dropout_probability=args.dropout,
                                                use_batch_norm=args.batch_norm,
                                                num_bins=args.num_spline,
                                                tails='linear',
                                                tail_bound=RQS_tail_bound
                                                )

        perm = nflows.transforms.permutations.Permutation(permutation_layer(args))    
        list_transforms.append(flow)
        list_transforms.append(perm)
    transform = nflows.transforms.base.CompositeTransform(list_transforms) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    model = nflows.flows.base.Flow(transform, base_dist)
    if save_architecture:
        with open(args.workdir+'/model_architecture.txt', 'w') as file: file.write('model = {}\n'.format(model))
    return model


def coupling_flow(args,
                  RQS_tail_bound=12, 
                  use_residual_blocks=None, 
                  random_mask=False, 
                  save_architecture=True):

    def resnet(in_features, out_features):
        return nflows.nn.nets.ResidualNet(
                                    in_features, out_features,
                                    context_features=args.dim_z,
                                    hidden_features=args.dim_hidden,
                                    num_blocks=args.num_blocks,
                                    activation=args.activation,
                                    dropout_probability=args.dropout,
                                    use_batch_norm=args.batch_norm
                                    )
    dim = args.dim_flow
    flow_model = args.flow + '_' + args.flow_func
    list_transforms = [] 
    mask = torch.ones(dim)
    if args.coupl_mask=='checkers': mask[::2]=-1
    elif args.coupl_mask=='mid-split': mask[int(dim/2):]=-1  # 2006.08545
    
    print('INFO: model -> {}, {} transform'.format(args.flow, args.flow_func))

    for _ in range(args.num_flows):   

        if flow_model=='coupling_RQSpline': 
            flow = nflows.transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
                                            mask=mask,
                                            transform_net_create_fn=resnet,
                                            num_bins=args.num_spline,
                                            tails='linear',
                                            tail_bound=RQS_tail_bound
                                            )
        perm = nflows.transforms.permutations.Permutation(permutation_layer(args))
        list_transforms.append(flow)
        list_transforms.append(perm)
    transform = nflows.transforms.base.CompositeTransform(list_transforms) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    model = nflows.flows.base.Flow(transform, base_dist)
    
    if save_architecture:
        with open(args.workdir+'/model_architecture.txt', 'w') as file: file.write('model = {}\n'.format(model))
   
    return model