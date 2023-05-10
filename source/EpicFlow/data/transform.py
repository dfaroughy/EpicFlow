import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from EpicFlow.utils.base import Id, logit, expit


def get_jet_data(jet_net_sample, transform_pt=None, remove_mask=False):

    if not transform_pt: transform_pt=Id
    jets=dict()
    for i in range(1+jet_net_sample.shape[1]): jets[i]=list()
    for jet in tqdm(jet_net_sample[:10000], desc="getting jet data"):
        jet_masked=[]
        jet_unmasked=[]
        for particle in jet:
            eta=particle[0]
            phi=particle[1]
            pt=transform_pt(particle[2])
            mask=particle[3]
            if mask: 
                jet_unmasked.append([eta, phi, pt])
            jet_masked.append([eta, phi, pt, mask])
        n=len(jet_unmasked)
        jets[n].append(jet_unmasked if remove_mask else jet_masked)
    return jets


def generate_jets(flow_model, 
                  context_model, 
                  num_samples, 
                  num_constituents, 
                  path, 
                  device):
    jets=[]
    with open(path, 'w') as file:
        for _ in tqdm(range(num_samples), desc="sampling {} jets".format(num_samples)):
            z_glob, z_loc, z_glob_in, z_loc_in = context_model.sample_context(size_global=1, size_local=num_constituents)
            z_context = torch.cat([z_glob.repeat(num_constituents,1),
                                   torch.flatten(z_loc, end_dim=1),
                                   z_glob_in.repeat(num_constituents,1),
                                   torch.flatten(z_loc_in, end_dim=1)],1)
            jet=flow_model.sample(num_samples=1, context=z_context.detach())     
            jet_pT_sorted=torch.stack(sorted(jet.reshape(num_constituents,3), key=lambda jet: jet[2], reverse=True)).detach().tolist()
            jets.append(jet_pT_sorted)
            for constituent in jet[0]:
                file.write("%s\t" % constituent.detach().tolist())
            file.write('\n')
    return jets

def get_particle_kinematics(data, transform_pt=None):

    eta={}; phi={}; pt={}

    if not transform_pt: transform_pt=Id
        
    eta['all']=list(np.concatenate([np.array(x).T[0] for x in data]))
    phi['all']=list(np.concatenate([np.array(x).T[1] for x in data]))
    pt['all']=[transform_pt(x) for x in list(np.concatenate([np.array(x).T[2] for x in data]))]
    multiplicity=[len(x) for x in data]

    for i in range(np.max(multiplicity)):
        eta[i]=[]; phi[i]=[]; pt[i]=[]

    for x in data:
        for i in range(np.max(multiplicity)):
            if i<=len(x)-1:
                eta[i].append(float(x[i][0]))
                phi[i].append(float(x[i][1]))
                pt[i].append(transform_pt(float(x[i][2])))

    return eta, phi, pt

def get_jet_kinematics(data, transform_pt=None):

    pt_rel={}; m_rel={}
    
    def Id(x):return x
    if not transform_pt: transform_pt=Id

    multiplicity=[len(x) for x in data]

    for i in range(np.max(multiplicity)+1):
        pt_rel[i]=[]; m_rel[i]=[]

    for jet in data:
        
        N=len(jet)
        jet_pt=0.; jet_px=0.; jet_py=0.; jet_pz=0.; jet_e=0.
        
        for particle in jet:
            
            theta=2.*np.arctan(np.exp(-float(particle[0])))
            phi=float(particle[1])
            e=transform_pt(float(particle[2])) / np.sin(theta)

            jet_e  += e
            jet_px += e * np.sin(theta) * np.cos(phi)
            jet_py += e * np.sin(theta) * np.sin(phi)
            jet_pz += e * np.cos(theta) 
            jet_pt += transform_pt(float(particle[2])) 

        pt_rel[N].append(jet_pt) 
        m_rel[N].append(np.sqrt(jet_e**2-jet_px**2-jet_py**2-jet_pz**2))
                                   
    return multiplicity, pt_rel, m_rel


class GaiaTransform:

    def __init__(self, data, covs, args):
        
        self.args = args
        self.data = torch.cat((data, covs), dim=1)
        self.mean = torch.zeros(6)
        self.std = torch.zeros(6)
        self.R = None

    @property
    def x(self):
        return self.data[:, :3]
    @property
    def v(self):
        return self.data[:, 3:6]
    @property
    def xv(self):
        return self.data[:, :6]
    @property
    def covs(self):
        return self.data[:,6:]
    @property
    def num_stars(self):
        return self.data.shape[0]

    def get_stars_near_sun(self, R=None, verbose=True):
        if not R: R=self.args.radius 
        if verbose: print('INFO: fetching stars within radius {} kpc from sun'.format(R))
        distance = torch.norm(self.data[:, :3] - torch.tensor(self.args.x_sun), dim=-1)
        self.data = self.data[ distance < R] 
        return self

    def smear(self, verbose=True):
        if verbose: print('INFO: smearing data')
        covs_matrix = torch.reshape(self.covs, (-1, 6, 6)) 
        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(self.num_stars, 6), covs_matrix)
        noise = noise_dist.sample()
        self.data[:,:6] = self.data[:,:6] + noise 
        return self

    def to_unit_ball(self, R=None, inverse=False, verbose=True): 
        x0 = torch.tensor(self.args.x_sun)
        if R:
            self.R = R
        else:
            dist = torch.norm(self.data[:,:3] - x0, dim=-1)
            self.R = torch.max(dist) * (1+1e-6)
        if verbose: print('INFO: centering and scaling to unit ball at origin, scale={}'.format(R))
        if inverse: 
            self.data[:,:3] = (self.x * self.R ) + x0 
        else:  
            self.data[:,:3] = (self.x - x0) / self.R
        return self

    def radial_blowup_transform(self, inverse=False, verbose=True):
        if verbose: print('INFO: transform hard edge of data to infinity')
        x_norm = torch.linalg.norm(self.x, dim=-1, keepdims=True)
        if inverse: 
            self.data[:,:3] = (self.x / x_norm) * torch.tanh(x_norm)
        else: 
            self.data[:,:3] =  (self.x / x_norm)  * torch.atanh(x_norm)
        return self

    def standardization(self, inverse=False, verbose=True):
        if verbose: print('INFO: standardizing data') 
        if inverse: 
            self.data[:,:3] = self.x * self.std[:3] + self.mean[:3]
            self.data[:,3:6] = self.v * self.std[3:] + self.mean[3:]
        else: 
            self.mean[:3] = torch.mean(self.data[:,:3], dim=0)
            self.mean[3:] = torch.mean(self.data[:,3:6], dim=0)
            self.std[:3] = torch.std(self.data[:,:3], dim=0)
            self.std[3:] = torch.std(self.data[:,3:6], dim=0)
            self.data[:,:3] = (self.x - self.mean[:3]) / self.std[:3]
            self.data[:,3:6] = (self.v - self.mean[3:]) / self.std[3:]
        return self 

    def preprocess(self, R=None, revert=False, verbose=True):  
        if verbose: print('INFO: preprocessing data')
        x0 = torch.tensor(self.args.x_sun)
        if R:
            self.R = R
        else:
            dist = torch.norm(self.data[:,:3] - x0, dim=-1)
            self.R = torch.max(dist) * (1+1e-6)

        if revert: 
            self.standardization(inverse=True, verbose=False)
            self.radial_blowup_transform(inverse=True, verbose=False)
            self.to_unit_ball(R=R, inverse=True, verbose=False)
        else: 
            self.to_unit_ball(R=R, verbose=False)
            self.radial_blowup_transform( verbose=False)
            self.standardization(verbose=False)
        return self 
