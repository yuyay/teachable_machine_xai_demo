"""Project-wide constants for the XAI demo.

Centralizes all tunable parameters (immutable module-level constants) so no
hyperparameters are hardcoded across modules.
"""

# Image preprocessing
IMG_SIZE: int = 224

# Integrated Gradients
M_STEPS: int = 32      # number of interpolation steps (was 50)
IG_BATCH: int = 8      # interpolation images processed per gradient batch

# TensorFlow CPU threading (instance has 2 vCPUs on Cloud Run)
TF_INTRA_THREADS: int = 2
TF_INTER_THREADS: int = 1

# In-instance concurrency guard for the heavy XAI computation
XAI_SEMAPHORE: int = 2

# Streamlit resource cache bounds for per-user uploaded models
MODEL_CACHE_MAX_ENTRIES: int = 8
MODEL_CACHE_TTL: int = 1800  # seconds

# Explanation methods
DEFAULT_XAI_METHOD: str = "Integrated Gradients"

# RISE (Randomized Input Sampling for Explanation)
RISE_N: int = 500        # number of random masks
RISE_GRID: int = 7       # low-res mask grid (s x s)
RISE_PROB: float = 0.5   # probability a grid cell is on (p1)
RISE_BATCH: int = 64     # masks per model forward batch (bounds peak memory)
RISE_SEED: int = 42      # reproducibility
