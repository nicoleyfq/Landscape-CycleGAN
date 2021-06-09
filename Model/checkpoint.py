import torch

def load_checkpoint(path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % path)
    return checkpoint

def save_checkpoint(state, save_path):
    torch.save(state, save_path)