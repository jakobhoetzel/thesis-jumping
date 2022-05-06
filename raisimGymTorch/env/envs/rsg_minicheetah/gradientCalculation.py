import numpy as np
import torch

def gradient_calculation(network, obs, obs_old):
    """ calculates gradient of output variable w.r.t. input variable"""

    # https://discuss.pytorch.org/t/gradient-of-output-wrt-specific-inputs/58585

    x = torch.zeros(1,network.input_shape[0])  # inputs to model (154)
    x[0][:] = torch.from_numpy(obs)  # Give all input and state data we have to the NN
    x.requires_grad = True  # Make sure gradients can be extracted

    # Compute outputs
    out = network.architecture(x)
    gradient = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=torch.ones_like(out),
                                   retain_graph=True)
    gradient = gradient[0].tolist()
    gradient = gradient[0]

    #old gradient
    if obs_old is not None:
        x = torch.zeros(1,network.input_shape[0])  # inputs to model (154)
        x[0][:] = torch.from_numpy(obs_old)  # Give all input and state data we have to the NN
        x.requires_grad = True  # Make sure gradients can be extracted

        # Compute outputs
        out = network.architecture(x)
        gradient_old = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=torch.ones_like(out),
                                       retain_graph=True)
        gradient_old = gradient_old[0].tolist()
        gradient_old = gradient_old[0]
    else:
        gradient_old = gradient.copy()
        obs_old = obs.copy()


    # rot_.e().row(2).transpose(), /// body orientation(z-axis in world frame expressed in body frame). 3 (0-2)
    # gc_.tail(12), /// joint angles 12 (3-14)
    # bodyAngularVel_, /// body angular velocity. 3 (15-17)
    # gv_.tail(12), /// joint velocity 12 (18-29)
    # previousAction_, /// previous action 12 (30-41)
    # prepreviousAction_, /// preprevious action 12 (42-53)
    # jointPosErrorHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 6) * nJoints_, nJoints_), /// joint History 24 (54-77)
    # jointPosErrorHist_.segment((historyLength_ - 4) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 4) * nJoints_, nJoints_), /// joint History 24 (78-101)
    # jointPosErrorHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), /// joint History 24 (102-125)
    # rot_.e().transpose() * (footPos_[0].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[1].e() - gc_.head(3)),
    # rot_.e().transpose() * (footPos_[2].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[3].e() - gc_.head(3)),
    #               /// relative foot position with respect to the body COM, expressed in the body frame 12 (126-137)
    # command_,  /// command 3 (138-140)
    # 0.0, gc_(0); //x_pos; sensor observation in environment 2 (141-142)
    # robot State (estimated) 12 (143-154)

    gradient = np.array(gradient)
    gradient_old = np.array(gradient_old)
    mean_grad = (gradient + gradient_old) / 2
    influence = np.transpose(mean_grad * (obs-obs_old))

    return influence

