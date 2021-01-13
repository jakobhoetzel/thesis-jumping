from shutil import copyfile
import datetime
import os
import ntpath
import torch


class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir
        

def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)


def load_param(file, env, actor, critic, optimizer):
    full_checkpoint_path = file.rsplit('/', 1)[0] + '/' + 'full_' + file.rsplit('/', 1)[1].split('_', 1)[1] + '.pt'
    env.load_scaling(file.rsplit('/', 1)[0], int(file.rsplit('/', 1)[1].split('_', 1)[1]))
    checkpoint = torch.load(full_checkpoint_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
