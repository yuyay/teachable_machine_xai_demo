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
