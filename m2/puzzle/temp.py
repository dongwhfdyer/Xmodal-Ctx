import time
from pathlib import Path

import torch
from torch import nn
from tqdm.contrib import tenumerate

from build_net import make_model_for_3_classes
from datasets import f1979_dataset
from models import resnet50_output_3
from args import args
from transforms import get_transforms
import torch.utils.data as data

from utils import get_optimizer, logger, AverageMeter, save_checkpoint


def train(train_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for i, (inputs, labels) in enumerate(train_loader): # batch_size * 3 * 512 * 512 ,batch_szie * 1
        inputs = inputs.cuda()
        labels = labels.cuda()
        # half precision: float32 -> float16
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(torch.sum(torch.argmax(outputs, dim=1) == labels).item() / inputs.size(0))
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(eopoch, i, len(train_loader), losses.avg, train_acc.avg))
    return losses.avg, train_acc.avg


def val(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    val_acc = AverageMeter()
    for i, (inputs, labels) in tenumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        val_acc.update(torch.sum(torch.argmax(outputs, dim=1) == labels).item() / inputs.size(0))
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(eopoch, i, len(val_loader), losses.avg, val_acc.avg))
    return losses.avg, val_acc.avg


if __name__ == '__main__':
    global best_acc
    best_acc = 0

    time_now = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    checkpoint_path = Path("runs/train/{}".format(time_now))

    train_trans = get_transforms(input_size=args.image_size, test_size=args.image_size)
    train_set = f1979_dataset(args.data_path, args.train_txt, train_trans["train"])
    val_set = f1979_dataset(args.data_path, args.val_txt, train_trans["test"])

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
    model = make_model_for_3_classes(args.model_path)
    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        model.cuda()
        criterion.cuda()

    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    for epoch in range(args.epochs):
        logger.info('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        test_loss, val_acc = val(val_loader, model, criterion)

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
