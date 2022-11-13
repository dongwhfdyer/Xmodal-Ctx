from einops import reduce
from torch import nn
import torchsummary
import torch
import torch.nn.functional as F


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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class puzzleSolver(nn.Module):
    def __init__(self):
        super(puzzleSolver, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.textSuperLongConv = nn.Conv2d(1, 1024, kernel_size=(3, 10199), stride=1, padding=1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(4096, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 81),
        #     nn.ReLU(),
        # )
        self.mlp = MLP(3072, 512, 81, 3)

    def forward(self, obj, caption):
        captionFeat = self.textSuperLongConv(caption[:, None, :, :])
        captionFeat_ = self.global_pool(captionFeat).squeeze()
        objFeat = reduce(obj, 'b h c -> b c', 'mean')
        # objFeat = reduce(obj, 'b h c -> b c', 'mean')[:, :2048] # kuhn: When the input is 2054, use this line.

        allFeature = torch.cat([captionFeat_, objFeat], dim=1)
        # ---------kkuhn-block------------------------------ # output the puzzle order
        cls = self.mlp(allFeature)
        cls = cls.view(cls.size(0), 9, 9)
        # ---------kkuhn-block------------------------------

        # #---------kkuhn-block------------------------------ # output the puzzle id
        # clsOneHot = self.mlp(allFeature)
        # cls = torch.argmax(clsOneHot, dim=1)
        # #---------kkuhn-block------------------------------

        return cls

        # idea
        # obj feature: [bs, bounding_box, 2048]
        # global pooling -> [bs, 2048]

        # caption: [bs, word_num, 10199]
        # conv kernel size: [3 * 10199] * 2048
        # conv ->: [bs, word_num-1, 2048]

        # obj + caption -> [bs, 4096]
        # MLP1 [4096, 512]
        # MLP2 [512, 128]
        # MLP3 [128, 64]


if __name__ == '__main__':
    torchsummary.summary(puzzleSolver.cuda(), (3, 608, 608))
