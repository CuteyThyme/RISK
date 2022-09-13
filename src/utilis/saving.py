import os
import shutil
import torch


def save_checkpoint(state, is_best, best_model_name):
    # savename = os.path.join(os.path.dirname(log_path), "epoch_" + str(epoch) + "_" + filename)
    # torch.save(state, savename)
    if is_best:
        torch.save(state, best_model_name)
