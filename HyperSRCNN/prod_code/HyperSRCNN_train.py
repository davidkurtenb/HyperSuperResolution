import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import datetime
import os

import sys
sys.path.append('/homes/dkurtenb/projects/hypersuperresolution/SRCNN/prod_code')
sys.path.append('/homes/dkurtenb/projects/hypersuperresolution/HyperSRCNN/prod_code')

from SRCNN_utils import *
from HyperSRCNN_utils import *

hyper_start_time = datetime.datetime.now()
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = '/homes/dkurtenb/projects/hypersuperresolution'

#################    DATA SELECT ##################################

####################################################### BeeMachine
train_data_dir = os.path.join(base_dir,'data/beemachine/images/train')
val_data_dir = os.path.join(base_dir,'data/beemachine/images/val')


####################################################### DF2k
#train_data_dir = os.path.join(base_dir,'data/DF2K/DF2K_train_HR')
#val_data_dir = os.path.join(base_dir,'data/DF2K/DF2K_valid_HR')

epochs = 250
compute = '4c2gpu'

if 'bee' in train_data_dir:
    results_dir = os.path.join(base_dir,f'HyperSRCNN/outputs/training_run_beemachine_{epochs}epochs_{compute}_{current_time}{current_time}')
elif 'DF2K' in train_data_dir:
    results_dir = os.path.join(base_dir,f'HyperSRCNN/outputs/training_run_df2k_{epochs}epochs_{compute}_{current_time}')
else:
    results_dir = os.path.join(base_dir,f'HyperSRCNN/outputs/training_run_{epochs}epochs_{compute}_{current_time}')
scale_factors = [2, 4, 8, 16]

train_datasets = {}
for scale in scale_factors:
    train_datasets[scale] = SRDataset(
        root_dir=train_data_dir,
        scale_factor=scale,
        patch_size=96,
        augment=True
    )

val_datasets = {}
for scale in scale_factors:
    val_datasets[scale] = SRDataset(
        root_dir=val_data_dir,
        scale_factor=scale,
        patch_size=96,
        augment=True
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnet = SRCNNNoWeights().to(device)
hnet = HyperSRCNN(num_models=len(scale_factors)).to(device)

#train
train_hyper_srcnn(
    hnet=hnet,
    mnet=mnet,
    train_datasets=train_datasets,
    val_datasets=val_datasets,
    scale_factors=scale_factors,
    results_dir=results_dir,
    batchsize=16,
    num_epochs=epochs,
    device=device,
    lr=0.0001
)

hyper_end_time = datetime.datetime.now()
hyper_training_time = hyper_end_time - hyper_start_time

print(f"\n{'='*50}")
print(f"HyperSRCNN Training Summary:")
print(f"{'='*50}")
print(f"Total Training Time for HyperSRCNN: {hyper_training_time}")

# Save the time to a file
with open(os.path.join(results_dir, 'hyper_training_time.txt'), 'w') as f:
    f.write(f"HyperSRCNN Training Time: {hyper_training_time}\n")