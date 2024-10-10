import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Convolutional Neural Network
# 1. Filter
# 2. Max Pooling
# 3. Fully Connected Layer

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

