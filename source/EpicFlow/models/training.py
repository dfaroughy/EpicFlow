
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from EpicFlow.models.loss import calculate_loss

def Train_Model(model, context, training_sample, validation_sample, args, show_plots=True, save_best_state=True):        
    
    train = Train_Epoch(model, context, args)
    test = Evaluate_Epoch(model, context, args)
        
    optimizers = { 'model' : torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5), 
                   'context': torch.optim.AdamW(context.parameters(), lr=args.lr, weight_decay=1e-5) }

    print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('INFO: number of context parameters: {}'.format(sum(p.numel() for p in context.parameters())))

    for epoch in tqdm(range(args.max_epochs), desc="epochs"):

        train.fit(training_sample, optimizers)       
        test.validate(validation_sample)

        print("\t Epoch: {}".format(epoch))
        print("\t Training loss: {}".format(train.loss))
        print("\t Test loss: {}  (min: {})".format(test.loss, test.loss_min))
    
        if test.check_patience(show_plots=show_plots, save_best_state=save_best_state): break

    plot_loss(train, test, args)
    torch.cuda.empty_cache()
    
    return test.best_model

#-------------------------------------------------

def sampler(model, num_samples, batch_size=10000):
    model.eval()
    with torch.no_grad(): 
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
        samples=[]
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            batch_samples = model.sample(num_samples=current_batch_size)
            samples.append(batch_samples)
        samples = torch.cat(samples, dim=0)
    return samples.cpu().detach()


#-------------------------------------------------


class Train_Epoch(nn.Module):

    def __init__(self, model, context, args):
        super(Train_Epoch, self).__init__()

        self.model = model
        self.context = context
        self.loss = 0
        self.loss_per_epoch = []
        self.args = args

    def fit(self, data, optimizer):

        self.model.train()
        self.context.train()
        self.loss = 0

        for batch in tqdm(data, desc="batch"):

            if self.args.batch_steps <= 1: 

                current_loss = calculate_loss(self.model, self.context, batch, self.args)
                current_loss.backward()
                optimizer['model'].step()
                optimizer['context'].step()  
                optimizer['model'].zero_grad()
                optimizer['context'].zero_grad()

                self.loss += current_loss.item() / len(data) 

            else: # sub-batch and accumulate gradient (use if data does not fit in GPU memory)  

                sub_batches = torch.tensor_split(batch, batch.shape[0] // self.args.batch_steps)
                sub_batch_loss = 0

                for sub_batch in sub_batches:
                    current_loss = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum)
                    current_loss.backward()
                    sub_batch_loss += current_loss.item() / self.args.batch_size

                optimizer['model'].step()
                optimizer['context'].step()  
                optimizer['model'].zero_grad()
                optimizer['context'].zero_grad()

                self.loss += sub_batch_loss / len(data) 

        self.loss_per_epoch.append(self.loss)

#-------------------------------------------------

class Evaluate_Epoch(nn.Module):

    def __init__(self, model, context,  args):
        super(Evaluate_Epoch, self).__init__()

        self.model = model
        self.context = context
        self.loss = 0
        self.loss_per_epoch = []
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.best_model = None
        self.terminate = False
        self.args = args

    def validate(self, data):

        self.model.eval()
        self.context.eval()
        self.loss = 0
        self.epoch += 1

        for batch in data:

            if self.args.batch_steps <= 1: 
            
                current_loss = calculate_loss(self.model, self.context, batch, self.args)
                self.loss += current_loss.item() / len(data)
        
            else:

                sub_batches = torch.tensor_split(batch, batch.shape[0] // self.args.batch_steps)
                sub_batch_loss = 0

                for sub_batch in sub_batches:
                    current_loss = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum) 
                    sub_batch_loss += current_loss.item() / self.args.batch_size

                self.loss += sub_batch_loss / len(data)

        self.loss_per_epoch.append(self.loss)


    def check_patience(self, show_plots=True, save_best_state=True):

        self.model.eval()

        if self.loss < self.loss_min:

            self.loss_min = self.loss
            self.patience = 0
            self.best_model = deepcopy(self.model)
            self.best_context = deepcopy(self.context)

            if show_plots:
                with torch.no_grad():
                    sample = sampler(self.best_model, num_samples=20000)
                    sample_x = sample[:,:3]
                    sample_v = sample[:,3:]
                    plot_data_projections(sample_x, bin_size=0.1, title=r'positions Epoch={}'.format(self.epoch), save=self.args.workdir  + '/results/result_x_Epoch_{}.pdf'.format(self.epoch))
                    plot_data_projections(sample_v, bin_size=0.2, title=r'velocities Epoch={}'.format(self.epoch), save=self.args.workdir  + '/results/result_v_Epoch_{}.pdf'.format(self.epoch),
                                                    xlim=[(-10, 10), (-10, 10), (-10, 10)], 
                                                    ylim=[(-10, 10), (-10, 10), (-10, 10)])
            if save_best_state:
                torch.save(self.best_model.state_dict(), self.args.workdir + '/best_model.pth')
                torch.save(self.best_context.state_dict(), self.args.workdir + '/best_context.pth')

        else: self.patience += 1
        if self.patience >= self.args.max_patience: self.terminate = True
    
        return self.terminate
