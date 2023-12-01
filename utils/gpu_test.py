import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# func to check if gpu is available
def gpu_check():
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")


if __name__ == "__main__":
    gpu_check()
