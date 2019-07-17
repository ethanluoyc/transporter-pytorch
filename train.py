import torch
from torch import nn
import numpy as np
import transporter
import torch.utils.data

batch_size = 64
image_channels = 3
k = 4

feature_encoder = transporter.FeatureEncoder(image_channels)
pose_regressor = transporter.PoseRegressor(image_channels, k)
refine_net = transporter.RefineNet(image_channels)

dataset = np.load('dataset.npz')
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dataset['image_t']).permute([0, 3, 1, 2]).clone().float(),
    torch.from_numpy(dataset['image_tp1']).permute([0, 3, 1, 2]).clone().float(),
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, pin_memory=True)


model = transporter.Transporter(
    feature_encoder, pose_regressor, refine_net
)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), 5e-4)

for num_epoch in range(10000):
    count = 0.0
    value = 0.0
    for xt, xtp1 in loader:
        xt = xt.cuda()
        xtp1 = xtp1.cuda()
        optimizer.zero_grad()
        reconstruction = model(xt, xtp1)
        loss = torch.nn.functional.mse_loss(reconstruction, xtp1)
        loss.backward()

        optimizer.step()
        value += loss.detach().item() * len(xt)
        count += len(xt)
    print("Epoch ", num_epoch, "Loss ", loss, "Avg Loss ", value / count)
    torch.save(model.state_dict(), "model.pth")
