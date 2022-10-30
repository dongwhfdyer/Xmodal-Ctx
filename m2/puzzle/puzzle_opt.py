import argparse

from easydict import EasyDict as edict

p_opt = edict()
# ---------kkuhn-block------------------------------ general config
p_opt.dataset_root = "datasets"
# p_opt.obj_file = "oscar.hdf5"
p_opt.obj_file = "puzzleCOCOFeature.hdf5"
p_opt.preload = False
p_opt.topk = 12
p_opt.start_epoch = 0
p_opt.epochs = 1000
p_opt.num_workers = 10
p_opt.model_path = "backup_models/resnet50-19c8e357.pth"
p_opt.save_prefix = "three_cls"
p_opt.use_cuda = True
p_opt.puzzle_file = "annotations/trainvalRandom9Info.txt"
p_opt.puzzle_id_mapping_file = "permutations_hamming_max_64.npy"
p_opt.seed = 1234
# ---------kkuhn-block------------------------------

p_opt.model_path = r"backup_models/resnet50-19c8e357.pth"
p_opt.batch_size = 200
p_opt.gpus = [0]
p_opt.lr = 0.0001
p_opt.resume = False


# rewrite the puzzle_opt.py file with parser.parse_args()
def get_args_parser():
    parser = argparse.ArgumentParser(description='Puzzle Solver')
    parser.add_argument('--dataset_root', default=p_opt.dataset_root, help='dataset root')
    parser.add_argument('--obj_file', default=p_opt.obj_file, help='object file')
    parser.add_argument('--preload', default=p_opt.preload, help='preload')
    parser.add_argument('--resume', default=p_opt.resume, help='resume')
    parser.add_argument('--topk', default=p_opt.topk, help='topk')
    parser.add_argument('--start_epoch', default=p_opt.start_epoch, help='start epoch')
    parser.add_argument('--epochs', default=p_opt.epochs, help='epochs')
    parser.add_argument('--num_workers', default=p_opt.num_workers, help='num workers')
    parser.add_argument('--model_path', default=p_opt.model_path, help='model path')
    parser.add_argument('--save_prefix', default=p_opt.save_prefix, help='save prefix')
    parser.add_argument('--use_cuda', default=p_opt.use_cuda, help='use cuda')
    parser.add_argument('--puzzle_file', default=p_opt.puzzle_file, help='puzzle file')
    parser.add_argument('--puzzle_id_mapping_file', default=p_opt.puzzle_id_mapping_file, help='puzzle id mapping file')
    parser.add_argument('--batch_size', default=p_opt.batch_size, help='batch size')
    parser.add_argument('--gpus', default=p_opt.gpus, help='gpus')
    parser.add_argument('--lr', default=p_opt.lr, help='lr')
    parser.add_argument('--seed', default=p_opt.seed, help='seed')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--pretrained', default="", help='use pre-trained model')
    opt = parser.parse_args()

    return opt
