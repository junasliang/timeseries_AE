import torch
from torchvision.models import AlexNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
epoch = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = CosineAnnealingLR(optimizer, T_max=epoch)
plt.figure()
x = list(range(epoch))
y = []

for epoch in range(1,epoch+1):
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    y.append(scheduler.get_last_lr()[0])

plt.plot(x,y)
plt.xlabel('epoch')
plt.ylabel('lr')
plt.savefig('lr_scheduler.jpg')