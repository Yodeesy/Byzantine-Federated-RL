import torch

from torch.nn import DataParallel

from matplotlib import animation
import matplotlib.pyplot as plt

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def resolve_env_name(base_name):
    """Resolve environment name across gymnasium versions.

    Tries common version suffixes (v2, v3, etc.) and returns the first
    available, falling back to the base name itself.
    """
    import gymnasium as _gym
    versions = ['v4', 'v3', 'v2', 'v1']
    for v in versions:
        name = f'{base_name}-{v}'
        try:
            _gym.envs.registration._find_spec(name)
            return name
        except Exception:
            continue
    return base_name  # last resort


def env_wrapper(name, obs):
    return obs

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=120)
    