import torch
import time
import os
import sys, platform


def check_gpu_avaiability():
    """
    Check if a GPU is available and print its details.
    """
    print("Python:", sys.executable)
    print("Platform:", platform.platform())
    print("Torch:", torch.__version__)
    print("Torch CUDA build:", torch.version.cuda)   # None == CPU-only wheel
    print("CUDA available:", torch.cuda.is_available())

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")