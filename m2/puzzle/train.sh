#torchrun --nproc_per_node=2 puzzle/train_puzzleSolver.py --batch_size 400 --resume True --resumeTime 11_01_23_08 --start_epoch 7
#CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 puzzle/train_puzzleSolver.py
#tensorboard --logdir=tensorboard_log
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 puzzle/train_puzzleSolver.py --batch_size 64 --pretrained runs/train/11_12_09_37/weights/model_cur.pth