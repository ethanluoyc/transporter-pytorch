import torch
from torch import nn, optim
import transporter
import data

batch_size = 32
image_channels = 1
k = 1

num_features = 32

feature_encoder = transporter.FeatureEncoder(image_channels, 3)
pose_regressor = transporter.PoseRegressor(image_channels, k, num_features)
refine_net = transporter.RefineNet(3, 1)

transporter = transporter.Transporter(
    feature_encoder, pose_regressor, refine_net
)

optimizer = torch.optim.Adam(transporter.parameters())


num_iterations = 1e5
for it in range(int(num_iterations)):
    xt, xtp1 = data.generate(batch_size)
    xt = torch.as_tensor(xt).unsqueeze(1)
    xtp1 = torch.as_tensor(xtp1).unsqueeze(1)

    optimizer.zero_grad()
    reconstruction = transporter(xt, xtp1)
    # loss = torch.nn.functional.mse_loss(reconstruction, xtp1)
    loss = torch.nn.functional.binary_cross_entropy(reconstruction, xtp1)
    loss.backward()

    optimizer.step()
    if it % 100 == 0:
        loss_mse = torch.nn.functional.mse_loss(reconstruction, xtp1).detach()
        print(it, loss.item(), loss_mse.item())