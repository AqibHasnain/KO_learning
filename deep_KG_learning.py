import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
print("Using PyTorch Version %s" %torch.__version__)
import pickle

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hl_sizes):
        super(Net, self).__init__()
        current_dim = input_dim
        self.linears = nn.ModuleList()
        for hl_dim in hl_sizes:
            self.linears.append(nn.Linear(current_dim, hl_dim))
            current_dim = hl_dim
        self.Lgen = nn.Parameter(torch.rand(output_dim,output_dim),requires_grad=True) # Koopman generator
        # self.L = nn.Parameter(torch.rand(output_dim,requires_grad=True,dtype=torch.cfloat)) # vector of eigenvalues
        # self.V = nn.Parameter(torch.rand(output_dim,output_dim,requires_grad=True,dtype=torch.cfloat)) # matrix of eigenvecs

    def forward(self, x):
        input_vecs = x
        for layer in self.linears:
            x = F.relu(layer(x))
        x = torch.cat((torch.Tensor(np.ones((x.shape[0],1))),input_vecs,x),dim=1)
        return x

if __name__ == '__main__':

    script_dir = os.path.dirname('deep_KG_learning.py') # getting relative path
    trained_models_path = os.path.join(script_dir, 'trained_models') # which relative path do you want to see
    data_path = os.path.join(script_dir,'data/')

    save_network = True
    net_name = '/slow_manifold_KG'

    save_trainLoss_fig = True
    figs_path = os.path.join(script_dir,'figures')

    ### Datasets ###

    dataset = 1 # each dataset contains the global snapshot matrix as well as the number of snapshots per trajectory, number of trajectories, and list of dt's
        
    if dataset == 0: 
        file_dir = 'toggle_switch_KG.p'

    if dataset == 1:
        file_dir = 'slow_manifold_KG.p'


    def get_snapshot_matrices(X,nT,nTraj): 
            '''This function assumes the global snapshot matrix is constructed with trajectories 
                sequentially placed in the columns'''
            prevInds = [x for x in range(0,nT-1)]
            forInds = [x for x in range(1,nT)]
            for i in range(0,nTraj-1):
                if i == 0:
                    more_prevInds = [x + nT for x in prevInds]
                    more_forInds = [x + nT for x in forInds]
                else: 
                    more_prevInds = [x + nT for x in more_prevInds]
                    more_forInds = [x + nT for x in more_forInds]
                prevInds = prevInds + more_prevInds
                forInds = forInds + more_forInds
            Xp = X[:,prevInds]
            Xf = X[:,forInds]
            return Xp,Xf
    
    X,nT,nTraj,dt_list = pickle.load(open(data_path+file_dir,'rb'))
    print(nT,nTraj)
    Xp,Xf = get_snapshot_matrices(X,nT,nTraj)
    trainXp = torch.Tensor(Xp.T)
    trainXf = torch.Tensor(Xf.T)
    testX = torch.Tensor(X.T)

    numDatapoints = nT*nTraj # number of total snapshots

    print('Dimension of the state: ' + str(trainXp.shape[1]));
    print('Number of trajectories: ' + str(nTraj));
    print('Number of total snapshots: ' + str(nT*nTraj));


    ### Neural network parameters ###

    NUM_INPUTS = trainXp.shape[1] # dimension of input
    NUM_HL = 4 # number of hidden layers (excludes the input layer)
    NODES_HL = 5   # number of nodes per hidden layer (number of learned observables)
    HL_SIZES = [NODES_HL for i in range(0,NUM_HL+1)] 
    NUM_OUTPUTS = NUM_INPUTS + HL_SIZES[-1] + 1 # output layer takes in dimension of input + 1 + dimension of hl's

    net = Net(NUM_INPUTS,NUM_OUTPUTS,HL_SIZES)
    print(net)

    ### Defining the loss function and the optimizer ###
    LEARNING_RATE = 0.0025 # an initially large learning rate will cause the eigvecs (net.V) to be ill-conditioned
    L2_REG = 0.0001
    MOMENTUM = 0.0

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=L2_REG)
    # optimizer = torch.optim.Adagrad(net.parameters(),lr=LEARNING_RATE,lr_decay=0.95,weight_decay=L2_REG)
    # look into optimizer which has built-in learning rate decay

    ### Training the network ###
    update_print_ct = 10
    epoch_to_save_net = 50
    lr_update = 0.9
    eps = 1e-15
    train_loss = []
    maxEpochs = 2000
    prev_loss = 0
    curr_loss = 1e10
    epoch = 0
    counteps = 0
    net.train()

    while (epoch <= maxEpochs): 

        if epoch % update_print_ct == 0:
            if np.abs(prev_loss - curr_loss) < eps:
                counteps += 1
                if counteps == 3:
                    print('The network has converged, eps = ' + str(eps))
                    break
            prev_loss = curr_loss

        for i in range(0,trainXp.shape[0]):

            dt = dt_list[i]

            K = torch.matrix_exp(net.Lgen*dt)
            # eL = torch.diag_embed(torch.exp(net.L*dt)) # exponential of the eigs, then embedded into a diagonal matrix
            # K = torch.matmul(torch.matmul(net.V,eL),torch.inverse(net.V)) # matrix representation of Koopman operator
            
            Kpsixp = torch.matmul(net(trainXp[i:i+1]),K) 
            psixf = net(trainXf[i:i+1])
            loss = loss_func(psixf, Kpsixp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        curr_loss = loss.item()

        if epoch % update_print_ct == 0:
            print('['+str(epoch)+']'+' loss = '+str(curr_loss))
            if curr_loss > prev_loss: # update learning rate
                for g in optimizer.param_groups:
                    g['lr'] = LEARNING_RATE * lr_update
                LEARNING_RATE = g['lr']
                print('Updated learning rate: ' + str(LEARNING_RATE))
            
        train_loss.append(loss.item()) 
        epoch+=1

        if epoch % epoch_to_save_net == 0:
            print('Saving network at epoch ' + str(epoch))
            if save_network:
                pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
                torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict


    print('['+str(epoch)+']'+' loss = '+ str(loss.item()))
    ### Done training ###

    ### Saving network (hyper)parameters ###
    if save_network:
        pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
        torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict

    ### Plotting the training loss ###
    import matplotlib.pyplot as plt;
    plt.rcParams.update({'font.size':14});
    plt.rcParams.update({'figure.autolayout': True})
    plt.semilogy(train_loss,lw=4);
    plt.ylabel('MSE Loss');
    plt.xlabel('Epoch');
    if save_trainLoss_fig:
        plt.savefig(figs_path+net_name+'_Loss.pdf')



