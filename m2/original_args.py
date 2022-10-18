import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
parser.add_argument('--exp_name', type=str, default='[m2][xmodal-ctx]')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--bs_reduct', type=int, default=5)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--m', type=int, default=40)
parser.add_argument('--topk', type=int, default=12)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--lr_xe', type=float, default=1e-4)
parser.add_argument('--lr_rl', type=float, default=5e-6)
parser.add_argument('--wd_rl', type=float, default=0.05)
parser.add_argument('--drop_rate', type=float, default=0.3)
parser.add_argument('--devices', nargs='+', type=int, default=[0, 1])
parser.add_argument('--dataset_root', type=str, default="./datasets")
parser.add_argument('--obj_file', type=str, default="oscar.hdf5")
parser.add_argument('--preload', action='store_true')
parser.add_argument('--resume_last', action='store_true')
parser.add_argument('--resume_best', action='store_true')
args = parser.parse_args()

