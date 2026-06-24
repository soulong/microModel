import re

import torch
import torch.nn as nn


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def can_compile(device):
    try:
        test = nn.Linear(1, 1).to(device)
        compiled = torch.compile(test)
        compiled(torch.randn(1, 1).to(device))
        del test, compiled
        return True
    except Exception:
        return False
