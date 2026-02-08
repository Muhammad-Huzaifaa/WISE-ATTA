import torch
import logging
import os, csv
import random
import numpy as np

logger = logging.getLogger(__name__)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update


def print_memory_info():
    logger.info('-' * 40)
    mem_dict = {}
    for metric in ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']:
        mem_dict[metric] = eval(f'torch.cuda.{metric}()')
        logger.info(f"{metric:>20s}: {mem_dict[metric] / 1e6:10.2f}MB")
    logger.info('-' * 40)
    return mem_dict


def set_deterministic(seed: int = 42):
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # CUDA libraries determinism (cuBLAS)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / matmul determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enforce deterministic ops (warn_only=True if you prefer warnings)
    torch.use_deterministic_algorithms(True, warn_only=False)


# utils/selection_stats.py

def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


class SelectionStatsLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "step", "batch_id",
                    "run_strategy",     # which strategy is used to adapt in THIS run
                    "criterion",        # which strategy produced this logged selection
                    "used_for_update",  # 1 if criterion == run_strategy else 0
                    "sel_idx",
                    "entropy_sel",
                    "drift_sel",
                ])

    def log(self, *, step, batch_id, run_strategy, criterion, used_for_update,
            sel_idx, entropy_sel, drift_sel):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, batch_id, run_strategy, criterion, used_for_update,
                        sel_idx, entropy_sel, drift_sel])

