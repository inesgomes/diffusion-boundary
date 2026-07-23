"""Package for the stress testing the diffusion boundary of a given classifier."""

import os

# cuBLAS is only reproducible with a fixed workspace, and torch reads this from the environment, so
# it has to be set before torch is imported. `python -m src` runs this file before src/__main__.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
