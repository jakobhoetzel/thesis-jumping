def freeze_actors(ppo):
    for param in ppo.actor_run.parameters():
        param.requires_grad = False
    for param in ppo.actor_jump.parameters():
        param.requires_grad = False
    for param in ppo.critic_run.parameters():
        param.requires_grad = False
    for param in ppo.critic_jump.parameters():
        param.requires_grad = False
    for param in ppo.actor_manager.parameters():
        param.requires_grad = True
    for param in ppo.critic_manager.parameters():
        param.requires_grad = True
    ppo.set_manager_training(True)


def freeze_manager(ppo):
    for param in ppo.actor_run.parameters():
        param.requires_grad = True
    for param in ppo.actor_jump.parameters():
        param.requires_grad = True
    for param in ppo.critic_run.parameters():
        param.requires_grad = True
    for param in ppo.critic_jump.parameters():
        param.requires_grad = True
    for param in ppo.actor_manager.parameters():
        param.requires_grad = False
    for param in ppo.critic_manager.parameters():
        param.requires_grad = False
    ppo.set_manager_training(False)
