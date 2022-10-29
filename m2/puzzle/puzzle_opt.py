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
# ---------kkuhn-block------------------------------

p_opt.model_path = r"backup_models/resnet50-19c8e357.pth"
p_opt.batch_size = 50
p_opt.gpus = [0]
p_opt.lr = 0.0001
