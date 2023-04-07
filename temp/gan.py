import torch
import torch.nn as nn
import torchvision


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape of x  [batch,1,28,28]
        out = self.model(x)
        out = out.reshape(x.shape[0], 1, 28, 28)
        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape of x  [batch,1,28,28]
        out = self.model(x.reshape(x.shape[0], -1))
        return out


transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(28),
     torchvision.transforms.ToTensor(),
     # torchvision.transforms.Normalize([0.5], [0.5])
     ]

)

batch_size = 64  # kuhn
dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

dim = 96
generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=150)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=150)
criterion = nn.BCELoss()  # kuhn
# criterion = nn.BCEWithLogitsLoss()

labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

use_gpu = torch.cuda.is_available()

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion = criterion.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

num_epoch = 100
for epoch in range(num_epoch):
    for i, data in enumerate(train_loader):
        img, _ = data
        # if last batch is not full, skip
        if img.shape[0] != batch_size:
            continue
        z = torch.randn([batch_size, dim])
        if use_gpu:
            z = z.to("cuda")
            img = img.to("cuda")

        pred_img = generator(z)
        dis = discriminator(pred_img)

        g_optimizer.zero_grad()
        recons_loss = torch.abs(pred_img - img).mean()
        generator_loss = recons_loss * 0.5 + criterion(labels_one, dis)
        generator_loss.backward()
        g_optimizer.step()
        scheduler_g.step()

        img_truth = discriminator(img)
        d_optimizer.zero_grad()
        real_loss = criterion(img_truth, labels_one)
        fake_loss = criterion(discriminator(pred_img.detach()), labels_zero)

        # discriminator_loss = criterion(labels_zero, discriminator(pred_img.detach())) + criterion(labels_one, img_truth)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        d_optimizer.step()
        scheduler_d.step()

        if (i % 400 == 0):
            print(f"epoch:{epoch} 第{i}轮, recons_loss:{recons_loss.item()}, "
                  f"g_loss:{generator_loss.item()}, d_loss:{discriminator_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

            # print(f'epoch:{epoch} 第{i}轮 g_loss:{generator_loss.item()} d_loss:{discriminator_loss.item()}')
            # torchvision.utils.save_image(pred_img, f'image_epoch{epoch}_{i}.png') # kuhn
            torchvision.utils.save_image(pred_img[:16].data, f'image_epoch{epoch}_{i}.png')
