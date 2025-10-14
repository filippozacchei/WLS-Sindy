# Imports
import numpy as np
import random
import os
import sys
import pysindy as ps

# Warnings
import warnings 
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
