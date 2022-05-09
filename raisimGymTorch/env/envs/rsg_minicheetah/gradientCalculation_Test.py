import numpy as np
import torch
from torch import nn
from random import random

class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the MLP
    mlp = MLP().to(device)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)
    loss_function = torch.nn.MSELoss()
    batchSize = 2
    current_loss = 0.0

    # Run the training loop
    for iteration in range(0, 30000):

        # Get and prepare inputs
        x = np.array([np.random.uniform(-1, 11, batchSize)], dtype=np.float32)
        x = np.reshape(x,[batchSize,1])
        x = torch.from_numpy(x).to(device)
        y = 2*x*x - 3*x - 2

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(x)

        # Compute loss
        loss = loss_function(outputs, y)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        num_print = 500
        if iteration % num_print == 0:
            if iteration == 0:
                print('Loss after iteration ', iteration, ': ', current_loss)
            else:
                print('Loss after iteration ', iteration, ': ', current_loss/num_print)
            current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')



    # https://discuss.pytorch.org/t/gradient-of-output-wrt-specific-inputs/58585
    gradient_old = None
    obs_old = None
    numSteps = 100
    grad_array = np.zeros([numSteps,1])
    influence_array = np.zeros([numSteps,1])
    y_array = np.zeros([numSteps,1])

    for step in range(0, numSteps):
        obs = np.array([step/numSteps * 10])
        if obs_old is None:
            obs_old = obs
        x = torch.zeros(1,1, device=device)  # inputs to model (154)
        x[0][:] = torch.from_numpy(obs)  # Give all input and state data we have to the NN
        x.requires_grad = True  # Make sure gradients can be extracted

        # Compute outputs
        out = mlp(x)
        gradient = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=torch.ones_like(out),
                                       retain_graph=True)
        gradient = gradient[0].tolist()
        gradient = gradient[0]

        #old gradient
        if gradient_old is None:
            gradient_old = gradient.copy()

        gradient = np.array(gradient)
        gradient_old = np.array(gradient_old)
        mean_grad = (gradient + gradient_old) / 2
        influence = np.transpose(mean_grad * (obs-obs_old))

        grad_array[step] = gradient
        influence_array[step] = influence
        y_array[step] = out.detach().cpu().numpy()

        gradient_old = gradient.copy()
        obs_old = obs

    print('-------- grad ---------')
    print(grad_array)
    print('-------- influence ---------')
    print(influence_array)
    print('-------- output ---------')
    print(y_array)
