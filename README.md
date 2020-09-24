# KO_learning

As of now, the code is set up for autonomous dynamical systems. Extending to controlled systems is future work. 

To add a new dataset, save a python list of the following in a pickle file: 

[Global snapshot matrix (numpy array), number of snapshots per trajectory, number of trajectories]

in the data directory. The code assumes that the global snapshot matrix is constructed with each trajectory adjacent to another and that each column is a snapshot of the state.

Now in deep_KO_learning.py 
- Update the datasets section with your new dataset
- Specify if you want to save the pytorch network with the save_network variable
- If saving the network, specify the net_name you would like

Run deep_KO_learning.py to train the network (update hyperparameters as necessary).

inference.py can be used for prediction, be sure to update the necessary variables and paths. 

# KG_learning

As of now, the code is set up for autonomous dynamical systems. Extending to controlled systems is future work. 

To add a new dataset, save a python list of the following in a pickle file: 

[Global snapshot matrix (numpy array), number of snapshots per trajectory, number of trajectories, list of timesteps]

in the data directory. The code assumes that the global snapshot matrix is constructed with each trajectory adjacent to another and that each column is a snapshot of the state.

deep_KG_learning_diagonalization is a working implementation, however it is restrictive in that it assumes the Koopman operator is diagonalizable (i.e. admits an eigendecomposition with no repeated eigenvalues). 



