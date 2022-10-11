from easydict import EasyDict as edict

p_opt = edict()
# ---------kkuhn-block------------------------------ general config
p_opt.dataset_root = "datasets"
p_opt.obj_file = "oscar.hdf5"
p_opt.preload = False
p_opt.topk = 12
p_opt.start_epoch = 0
p_opt.epochs = 1000
p_opt.num_workers = 10
p_opt.model_path = "backup_models/resnet50-19c8e357.pth"
p_opt.save_prefix = "three_cls"
p_opt.use_cuda = True
# ---------kkuhn-block------------------------------

p_opt.model_path = r"backup_models/resnet50-19c8e357.pth"


class puzzleOpt:
    def __init__(self):
        self.data_path = "datasets/f1979/images"
