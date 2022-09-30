import argparse
from train import TrainUNIT

# Arguments
parser = argparse.ArgumentParser(description='Train CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--offset_epochs', type=int, default=0)
parser.add_argument('--decay_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)

# Weighted Loss
parser.add_argument('--lambda0', type=float, default=10.0)
parser.add_argument('--lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', type=float, default=100.0)
parser.add_argument('--lambda3', type=float, default=0.1)
parser.add_argument('--lambda4', type=float, default=100.0)

# Model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--n_sample', type=int, default=2)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_res_blocks', type=int, default=3)

# Dataset
parser.add_argument('--train_size', type=int, default=3000)
parser.add_argument('--domain1', type=str, default='Dog')
parser.add_argument('--domain2', type=str, default='Cat')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()

train_UNIT = TrainUNIT(args)
train_UNIT.train()
