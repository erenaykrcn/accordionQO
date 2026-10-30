python - <<'PY'
import inspect, hashlib
from torchgpe.bec2D.potentials import DispersiveCavity

src = inspect.getsource(DispersiveCavity)
print(hashlib.sha256(src.encode()).hexdigest())
PY

python - <<'PY'
import torch, numpy as np, scipy
import sys

print("python", sys.version)
print("torch", torch.__version__)
print("numpy", np.__version__)
print("scipy", scipy.__version__)

print("torch file", torch.__file__)
print("threads", torch.get_num_threads())
print("interop", torch.get_num_interop_threads())
print("default dtype", torch.get_default_dtype())

print(torch.__config__.show())
PY
