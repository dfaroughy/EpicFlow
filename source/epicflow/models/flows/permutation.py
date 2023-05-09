import torch
import numpy as np

def permutation_layer(args):
    dim=args.dim
    if args.permutation:
        if 'cycle' in args.permutation:
            n=int(args.permutation.split('-')[0]) 
            if n<dim:
                p=list(range(dim)[-n:])+list(range(dim)[:-n]) # 3-cyclic : [0,1,2,3,4,5,6,7,8,9] -> [7,8,9,0,1,2,3,4,5,6]    
            else:
                raise ValueError('n-cycle must be a positive integer smaller than dim')
        elif args.permutation=='inverse':
            p=list(range(dim))
            p.reverse()
        else:
            raise ValueError('wrong permutation arg. Use [n]-cycle or inverse')
    else:
        p=list(range(dim))
    return torch.tensor(p)