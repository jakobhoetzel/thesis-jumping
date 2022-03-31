import numpy as np
import torch
from adamp import AdamP


def identity_learning(actor_manager, obs, guideline, device):
    """ Trains untrained actor network with imitation learning to select jump network close to hurdle """

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = AdamP(actor_manager.parameters(), lr=5e-4) #5e-4

    action_probs = actor_manager.architecture.architecture(torch.from_numpy(obs).to(device))

    # Compute and print loss
    loss = criterion(torch.from_numpy(guideline.astype('f')).to(device), action_probs[:,1:])
    # print("loss: ", loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
