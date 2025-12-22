# cuDNN library path fix - set before any PyTorch imports
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

# PyTorch 2.6 compatibility patch - MUST be first thing that runs
# This patches torch.load to use weights_only=False by default for trusted PyAnnote models
import torch

_original_load = torch.load

def _safe_load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **pickle_load_args):
    """Wrapper that defaults weights_only to False for PyAnnote model compatibility."""
    if weights_only is None:
        weights_only = False
    return _original_load(f, map_location=map_location, pickle_module=pickle_module, 
                         weights_only=weights_only, mmap=mmap, **pickle_load_args)

# Patch torch.load globally
torch.load = _safe_load

# Also patch the reference in torch.serialization module
import torch.serialization
torch.serialization.load = _safe_load
