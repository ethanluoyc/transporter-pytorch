import torch
from torch import nn
import numpy as np
import transporter
import torch.utils.data
from data import Dataset, Sampler
from torchvision import transforms

batch_size = 64
image_channels = 3
k = 4

feature_encoder = transporter.FeatureEncoder(image_channels)
pose_regressor = transporter.PoseRegressor(image_channels, k)
refine_net = transporter.RefineNet(image_channels)

transform = transforms.ToTensor()

dataset = Dataset('data', transform=transform)
sampler = Sampler(dataset)
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=sampler, pin_memory=True
)

model = transporter.Transporter(
    feature_encoder, pose_regressor, refine_net
)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    int(1e5),
    gamma=0.95, last_epoch=-1)

model.train()
for itr, (xt, xtp1) in enumerate(loader):
    xt = xt.cuda()
    xtp1 = xtp1.cuda()
    optimizer.zero_grad()
    reconstruction = model(xt, xtp1)
    loss = torch.nn.functional.mse_loss(reconstruction, xtp1)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if itr % 100 == 0:
        print("Itr ", itr, "Loss ", loss)
        torch.save(model.state_dict(), "model.pth")
    if itr > 1e6:
        break
