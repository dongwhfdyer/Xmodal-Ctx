import logging
import os

import torch


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(state, is_best, checkpoint):
    fold = ''
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best:
        model_name = 'model_' + str(state['epoch']) + '_' + str(int(round(state['train_acc'] * 100, 0))) + '_' + str(int(round(state['acc'] * 100, 0))) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)
        logger.info("update best model")


def advancedLogger():
    logger_name = "global"
    log_file = "logs/global.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    FORMATTER_1 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    FORMATTER_2 = logging.Formatter("[%(levelname)s] - %(filename)s - %(lineno)d: %(message)s")
    FORMATTER_3 = logging.Formatter("%(message)s")
    FORMATTER_4 = logging.Formatter("[%(levelname)s] - %(asctime)s - %(filename)s - %(lineno)d: %(message)s")

    ch.setFormatter(FORMATTER_3)
    fh.setFormatter(FORMATTER_4)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = advancedLogger()
