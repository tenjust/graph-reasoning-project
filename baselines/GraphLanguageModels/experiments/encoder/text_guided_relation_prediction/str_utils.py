import logging
from pathlib import Path
from typing import Optional

from torch import nn
from torch_geometric.graphgym import optim


def str2bool(s:str)->bool:
    # input can also be bool
    if s in ['True', 'true', '1', True]:
        return True
    elif s in ['False', 'false', '0', False]:
        return False
    elif s in ['None', None]:
        return None
    else:
        raise ValueError(f"unknown boolean value {s}")


def str2path(s:str)->Optional[Path]:
    if s in ['None', None]:
        return None
    else:
        return Path(s)


def str2int(s:str)->Optional[int]:
    if s in ['None', None]:
        return None
    else:
        return int(float(s))

def str2optimizer(s:str)->optim.Optimizer:
    if s == "Adam":
        return optim.Adam
    elif s == "SGD":
        return optim.SGD
    elif s == "AdamW":
        return optim.AdamW
    else:
        raise ValueError(f"unknown optimizer {s}")


def str2criterion(s:str)->nn.Module:
    if s == "CrossEntropyLoss":
        return nn.CrossEntropyLoss
    else:
        raise ValueError(f"unknown criterion {s}")


def str2logging_level(s:str):
    if s == "CRITICAL":
        return logging.CRITICAL
    elif s == "WARNING":
        return logging.WARNING
    elif s == "INFO":
        return logging.INFO
    elif s == "DEBUG":
        return logging.DEBUG
    else:
        raise ValueError(f"unknown logging_level {s}")
