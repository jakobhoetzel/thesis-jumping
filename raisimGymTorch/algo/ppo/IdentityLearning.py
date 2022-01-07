import numpy as np
import torch
from adamp import AdamP


def identity_learning(num_iterations, actor, act_dim, device):

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = AdamP(actor.parameters(), lr=5e-4)

    for t in range(num_iterations + 0):

        train_input = np.concatenate((np.random.normal(0., 4., size=(1,act_dim)), np.random.normal(0.0, 0.2 , size=(1,1)), np.random.normal(10., 2., size=(1,1))), axis=1, dtype=np.float32)
        train_action = actor.architecture.architecture(torch.from_numpy(train_input).to(device))

        # Compute and print loss
        loss = criterion(train_action, torch.from_numpy(train_input[:,0:act_dim]).to(device))
        if t % 2500 == 0 and t != 0:
            print("step: ", t, " loss: " , loss.item())
        if t % 25000 == 0 and t != 0:
            print("in: ", train_input)
            print("out: ", train_action)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
