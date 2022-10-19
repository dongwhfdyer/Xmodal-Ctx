from torch import nn
import torch


class cls_head(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cls_head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_dim, n_class)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class puzzleSolver(nn.Module):
    def __init__(self):
        super(puzzleSolver, self).__init__()

    def forward(self, obj, caption, puzzle):
        dd = caption[0].unsqueeze(1)
        batch_size = dd.shape[0]
        class_num = 10199
        y_one_hot = torch.zeros(batch_size, class_num).to("cuda").scatter_(1, dd, 1)  # kuhn: specify the device

        pass
