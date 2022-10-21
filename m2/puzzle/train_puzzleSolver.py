# Create the dataset
import pickle
import random
import time

import numpy as np
import torch
from torch import nn

from data import ImageDetectionsField, TextField, RawField, DataLoader
from data.field import puzzleIdField, OnehotTextField
from puzzle.puzzle_model import puzzleSolver
from puzzle_opt import p_opt
from data.dataset import PuzzleCOCO
# import Path
from pathlib import Path
from puzzle_utils import logger, AverageMeter


def prepareField():
    object_field = ImageDetectionsField(obj_file=Path(p_opt.dataset_root) / p_opt.obj_file, max_detections=50, preload=p_opt.preload)
    # text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    text_field = OnehotTextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    vocab_file = 'vocab/vocab_coco.pkl'
    text_field.vocab = pickle.load(open(vocab_file, 'rb'))
    puzzle_field = puzzleIdField(puzzleFile=datasetRoot / p_opt.puzzle_file, )
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

def getOneItemV2(data):
    obj = data["object"].to(cudaDevice, non_blocking=True)
    puzzle = data["puzzle_id"].to(cudaDevice, non_blocking=True)  # kuhn: what's non_blocking?
    captions = data["text"].to(cudaDevice, non_blocking=True)
    return obj, puzzle, captions


def build_model():
    return puzzleSolver().to(cudaDevice)


def initSeed():
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)


def train(train_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for i, data in enumerate(train_loader):  # batch_size * 3 * 512 * 512 ,batch_szie * 1
        obj, puzzle, captions = getOneItemV2(data)
        out = model(obj, puzzle, captions)
        loss = criterion(out, puzzle)


        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        update(loss, optimizer)
        losses.update(loss.item(), obj.size(0))
        train_acc.update(torch.sum(torch.argmax(outputs, dim=1) == labels).item() / inputs.size(0))
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(eopoch, i, len(train_loader), losses.avg, train_acc.avg))
    return losses.avg, train_acc.avg


def update(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # ---------kkuhn-block------------------------------ # basic setup
    global best_acc
    best_acc = 0
    initSeed()
    cudaDevice = "cuda:0"
    time_now = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    checkpoint_path = Path("runs/train/{}".format(time_now))
    # ---------kkuhn-block------------------------------

    # ---------kkuhn-block------------------------------ # dataloader setup
    datasetRoot = Path(p_opt.dataset_root)
    object_field, text_field, puzzle_field, fields = prepareField()

    annoFolder = datasetRoot / "annotations"
    puzzlecoco = PuzzleCOCO(fields, annoFolder, annoFolder)

    train_dataset, val_dataset, test_dataset = puzzlecoco.splits
    train_dataloader = DataLoader(train_dataset, batch_size=p_opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=p_opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # ---------kkuhn-block------------------------------

    # ---------kkuhn-block------------------------------ # criterion, model, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=p_opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    # ---------kkuhn-block------------------------------

    for epoch in range(p_opt.epochs):
        logger.info('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, p_opt.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        test_loss, val_acc = val(val_dataloader, model, criterion)

        scheduler.step()
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'fold': 0,
            'epoch': epoch + 1,
            'model': model,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

    # # ---------kkuhn-block------------------------------ # kuhn: only for testing
    # obj, puzzle, captions = genOneItem(train_dataloader)
    # out = model(obj=obj, puzzle=puzzle, caption=captions)
    # # ---------kkuhn-block------------------------------

    pass
