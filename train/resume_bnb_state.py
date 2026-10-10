"""Resume with restored paged-state metadata removed; leave installed packages intact."""
import runpy
import sys

import torch
from bitsandbytes.optim.optimizer import Optimizer8bit

original_load = Optimizer8bit.load_state_dict


def load_state(self, state_dict, *args, **kwargs):
    count = 0

    def convert(value):
        nonlocal count
        if isinstance(value, torch.Tensor):
            if getattr(value, "is_paged", False):
                # Allocate real ordinary storage rather than trust serialized UVM tags.
                result = torch.empty_like(value)
                result.copy_(value)
                for name in ("is_paged", "page_deviceid"):
                    if hasattr(result, name):
                        delattr(result, name)
                count += 1
                return result
            return value
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        return value

    cleaned = convert(state_dict)
    print(f"BNB resume: materialized {count} tensors with paged metadata", flush=True)
    return original_load(self, cleaned, *args, **kwargs)


Optimizer8bit.load_state_dict = load_state
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
