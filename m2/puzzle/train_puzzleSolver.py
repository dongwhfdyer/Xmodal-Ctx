# Create the dataset
import argparse
import json
import sys
import os
import pickle
import random
from hashlib import md5

import guli
import time

import yaml
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from tqdm.contrib import tenumerate
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import ImageDetectionsField, TextField, RawField, DataLoader
from data.field import puzzleIdField, OnehotTextField
from distributed_utils import get_rank, init_distributed_mode, is_main_process, reduce_value
from puzzle_model import puzzleSolver
from puzzle_opt import get_args_parser
from data.dataset import PuzzleCOCO
from pathlib import Path
from puzzle_utils import logger, AverageMeter, save_checkpoint


def prepareField():
    object_field = ImageDetectionsField(obj_file=Path(p_opt.dataset_root) / p_opt.obj_file, max_detections=50, preload=p_opt.preload)
    # text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    text_field = OnehotTextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    vocab_file = 'vocab/vocab_coco.pkl'
    text_field.vocab = pickle.load(open(vocab_file, 'rb'))
    puzzle_field = puzzleIdField(puzzleFile=datasetRoot / p_opt.puzzle_file, puzzleIdMappingFile=datasetRoot / p_opt.puzzle_id_mapping_file)
    fields = {
        "object": object_field,
        "text": text_field,
        "img_id": RawField(),
        "puzzle_id": puzzle_field,
    }
    return object_field, text_field, puzzle_field, fields


def genOneItem(dataloader):
    data = dataloader.__iter__().__next__()
    obj = data["object"].to(cudaDevice, non_blocking=True)
    puzzle = data["puzzle_id"].to(cudaDevice, non_blocking=True)  # kuhn: what's non_blocking?
    captions = data["text"].to(cudaDevice, non_blocking=True)
    return obj, puzzle, captions


def getOneItemV2(data, device):
    obj = data["object"].to(device, non_blocking=True)
    puzzle = data["puzzle_id"].to(device, non_blocking=True)  # kuhn: what's non_blocking?
    captions = data["text"].to(device, non_blocking=True)
    return obj, puzzle, captions


def build_model(device, pth_path):
    model = puzzleSolver().to(device)
    if pth_path is None:
        checkpoint_path = Path(guli.GuliVariable("checkpoint_path").get())
        pth_path = checkpoint_path / "init.pth"
        if is_main_process():
            torch.save(model.state_dict(), pth_path)
    torch.distributed.barrier()
    print("torch.device:", device, "pth_path:", pth_path)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    return model


def initSeed():
    seed = p_opt.seed + get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()
    # todo: make sure that all data go to gpu.
    if is_main_process():
        train_loader = tqdm(train_loader, file=sys.stdout, position=0, leave=True, ncols=100)

    for i, data in enumerate(train_loader):  # batch_size * 3 * 512 * 512 ,batch_szie * 1
        obj, puzzle, captions = getOneItemV2(data, device)
        # #---------kkuhn-block------------------------------ # kuhn: test dataloader
        # puzzleFirstCol = puzzle[:, 0]
        # hash = md5(puzzleFirstCol.cpu().numpy().tobytes())
        # print("hash:", hash.hexdigest(), "Device:", torch.cuda.current_device())
        # #---------kkuhn-block------------------------------
        out = model(obj=obj, caption=captions)  # todo: if it is too slow, unwrap these function. and delete the puzzle field in model definition
        loss = criterion(out, puzzle)  # todo: try half precision mode.
        loss = reduce_value(loss, average=True)

        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        update(loss, optimizer)
        losses.update(loss.item(), obj.size(0))
        train_acc.update(torch.sum(torch.argmax(out, dim=2) == puzzle).item() / (puzzle.shape[0] * puzzle.shape[1]))
        if is_main_process():
            train_loader.desc = "Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(epoch, i, len(train_loader), losses.avg, train_acc.avg)

        # # ---------kkuhn-block------------------------------ # kuhn: only for debug
        # ss = torch.argmax(out, dim=2)
        # kk = ss == puzzle
        # ll = torch.sum(kk)
        # # ---------kkuhn-block------------------------------
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(epoch, i, len(train_loader), losses.avg, train_acc.avg))
        # # ---------kkuhn-block------------------------------ # kuhn: only for debug
        # if i == 5:
        #     break
        # # ---------kkuhn-block------------------------------

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return losses.avg, train_acc.avg


@torch.no_grad()
def val(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    val_acc = AverageMeter()
    if is_main_process():
        val_loader = tqdm(val_loader, file=sys.stdout)
    for i, data in enumerate(val_loader):  # batch_size * 3 * 512 * 512 ,batch_szie * 1
        obj, puzzle, captions = getOneItemV2(data, device)
        out = model(obj=obj, caption=captions)
        loss = criterion(out, puzzle)
        loss = reduce_value(loss, average=True)
        losses.update(loss.item(), obj.size(0))
        val_acc.update(torch.sum(torch.argmax(out, dim=2) == puzzle).item() / (puzzle.shape[0] * puzzle.shape[1]))
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(eopoch, i, len(val_loader), losses.avg, val_acc.avg))
        # # ---------kkuhn-block------------------------------ # kuhn: only for debug
        # if i == 5:
        #     break
        # # ---------kkuhn-block------------------------------
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return losses.avg, val_acc.avg


def update(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_model_path(p_opt):
    if p_opt.resume:
        return checkpoint_path / "model_cur.pth"
    elif p_opt.pretrained != "":
        return Path(p_opt.pretrained)
    else:
        return None


if __name__ == '__main__':
    p_opt = get_args_parser()

    init_distributed_mode(args=p_opt)
    rank = p_opt.rank
    device = torch.device(p_opt.gpu)
    batch_size = p_opt.batch_size
    p_opt.lr *= p_opt.world_size
    initSeed()

    # ---------kkuhn-block------------------------------ # basic setup
    global best_acc
    best_acc = 0
    cudaDevice = "cuda:1"
    if is_main_process():
        time_now = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
        tensorboard_log_path = Path("tensorboard_log") / time_now
        checkpoint_path = Path("runs/train/{}".format(time_now))
        if p_opt.resume:
            checkpoint_path = Path("runs/train/{}".format(p_opt.resumeTime))
            tensorboard_log_path = Path("tensorboard_log") / p_opt.resumeTime
        guli.GuliVariable("checkpoint_path").setValue(str(checkpoint_path))

        writer = SummaryWriter(str(tensorboard_log_path))
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        with open(str(checkpoint_path / "config.yaml"), "w") as f:
            yaml.dump(vars(p_opt), f)
    # ---------kkuhn-block------------------------------

    # ---------kkuhn-block------------------------------ # dataloader setup
    datasetRoot = Path(p_opt.dataset_root)
    object_field, text_field, puzzle_field, fields = prepareField()

    annoFolder = datasetRoot / "annotations"
    puzzlecoco = PuzzleCOCO(fields, annoFolder, annoFolder)
    # train_dataset = puzzlecoco.get_train_dataset_only

    train_dataset, val_dataset, test_dataset = puzzlecoco.splits

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # kuhn: test nw
    print("Using {} dataloader workers every process".format(nw))

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=nw)  # kuhn: test pin_memory
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=nw, pin_memory=True, drop_last=True)
    # ---------kkuhn-block------------------------------

    # ---------kkuhn-block------------------------------ # criterion, model, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    model_path = get_model_path(p_opt)
    model = build_model(device, model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    if p_opt.resume:
        checkpoint = torch.load(checkpoint_path / "checkpoint.pth.tar", map_location='cpu')
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            p_opt.start_epoch = checkpoint['epoch'] + 1
    # ---------kkuhn-block------------------------------

    for epoch in range(p_opt.start_epoch, p_opt.epochs):
        train_sampler.set_epoch(epoch)
        logger.info('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, p_opt.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device, epoch)
        if epoch == 3:
            exit()
        else:
            continue
        test_loss, val_acc = val(val_dataloader, model, criterion, device)
        if is_main_process():
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', test_loss, epoch)
            writer.add_scalar("val_acc", val_acc, epoch)
        scheduler.step()
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        if is_main_process():
            save_checkpoint({
                'fold': 0,
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'train_acc': train_acc,
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
            }, is_best, checkpoint=checkpoint_path)

    # # ---------kkuhn-block------------------------------ # kuhn: only for testing
    # obj, puzzle, captions = genOneItem(train_dataloader)
    # out = model(obj=obj, puzzle=puzzle, caption=captions)
    # # ---------kkuhn-block------------------------------

    pass
