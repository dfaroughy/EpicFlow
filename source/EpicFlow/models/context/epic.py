import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import functional as F

class Epic_Layer(nn.Module):
    
    def __init__(self, args):
        super(Epic_Layer, self).__init__()
        self.device = args.device
        self.pooling = args.pooling
        self.dim_hidden = args.dim_hidden_z
        self.dim_glob = args.dim_glob_z
        self.dim_loc = args.dim_loc_z
               
        if self.pooling=='sum':
            self.phi_1 = nn.Sequential(
                                weight_norm(nn.Linear( self.dim_hidden + self.dim_glob, self.dim_hidden)),
                                nn.LeakyReLU()
                            ) 

        elif self.pooling=='sum_mean':
            self.phi_1 = nn.Sequential(
                                weight_norm(nn.Linear(int(self.dim_glob + 2*self.dim_hidden), self.dim_hidden)),
                                nn.LeakyReLU()
                            )
        self.phi_2 = weight_norm(nn.Linear(self.dim_hidden, self.dim_glob))   

        self.rho_1 = nn.Sequential(
                                weight_norm(nn.Linear(self.dim_hidden + self.dim_glob, self.dim_hidden)),
                                nn.LeakyReLU()
                            )

        self.rho_2 = weight_norm(nn.Linear(self.dim_hidden, self.dim_hidden)) 
        
    def pooling_function(self, z_glob, z_loc):   
        
        if self.pooling == 'sum':
            z_sum_pool = z_loc.sum(dim=1, keepdim=False)
            z_pool = torch.cat([z_glob, z_sum_pool], dim=1) 
            return z_pool

        if self.pooling == 'sum_mean':
            z_sum_pool = z_loc.sum(dim=1, keepdim=False)
            z_mean_pool = z_loc.mean(dim=1, keepdim=False)
            z_pool = torch.cat([z_glob, z_sum_pool, z_mean_pool], dim=1)  
            return z_pool # dim = 2*dim_loc + dim_glob  

    def forward(self, z_glob, z_loc): # z_glob = [batch, dim] , z_loc = [batch, n_points, dim]
        
        # local -> global        
        z_loc2glob = self.pooling_function(z_glob, z_loc)                     
        z_loc2glob = self.phi_1(z_loc2glob)                               
        z_glob = F.leaky_relu(self.phi_2(z_loc2glob) + z_glob)       
        
        # global -> local
        z_glob2loc = torch.unsqueeze(z_glob, dim=1).repeat(1,z_loc.size(1),1)    
        z_glob2loc = torch.cat([z_loc, z_glob2loc], 2)
        z_glob2loc = self.rho_1(z_glob2loc)   
        z_loc = F.leaky_relu(self.rho_2(z_glob2loc) + z_loc)

        return z_glob, z_loc
    

class Epic(nn.Module):
    
    def __init__(self, args):
        
        super(Epic, self).__init__()  

        self.args = args
        self.device = args.device
        self.pooling = args.pooling
        self.dim_hidden = args.dim_hidden_z
        self.dim_glob = args.dim_glob_z
        self.dim_loc = args.dim_loc_z
        self.num_layers = args.num_epic_layers
                    
        self.phi_loc = nn.Sequential(
                                    weight_norm(nn.Linear(self.dim_loc, self.dim_hidden )),
                                    nn.LeakyReLU()
                                    ) 

        self.phi_glob = nn.Sequential(
                                    weight_norm(nn.Linear(self.dim_glob, self.dim_hidden)),
                                    nn.LeakyReLU(),
                                    weight_norm(nn.Linear(self.dim_hidden, self.dim_glob)),
                                    nn.LeakyReLU()
                                    )  

        self.epic_list = nn.ModuleList()

        for _ in range(self.num_layers):
            self.epic_list.append(Epic_Layer(self.args))

        self.rho_loc = nn.Sequential(weight_norm(nn.Linear(self.dim_hidden, self.dim_loc)), nn.LeakyReLU()) 

    def forward(self, z_glob, z_loc):  # z_glob = [batch, dim] , z_loc = [batch, n_points, dim]
        z_loc = self.phi_loc(z_loc)      
        z_glob = self.phi_glob(z_glob)                        
        z_glob_in, z_loc_in = z_glob.clone(), z_loc.clone()
        for i in range(self.num_layers):
            z_glob, z_loc = self.epic_list[i](z_glob, z_loc)            # contains residual connection
            z_glob, z_loc = z_glob + z_glob_in, z_loc + z_loc_in        # skip connection to sampled input
        z_loc = self.rho_loc(z_loc)
        return z_glob, z_loc

    
    def sample(self, size_glob, size_loc):
        z_glob = torch.randn(size_glob, self.dim_glob).to(self.device)             # latent global parameter z ~ N(0,1)
        z_loc = torch.randn(size_glob, size_loc, self.dim_loc).to(self.device)     # latent local parameter z_i ~ N(0,1)
        z_glob_out, z_loc_out = self.forward(z_glob, z_loc)
        return  z_glob_out, z_loc_out, z_glob, z_loc


