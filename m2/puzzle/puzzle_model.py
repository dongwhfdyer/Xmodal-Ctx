from torch import nn

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
