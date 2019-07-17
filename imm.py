import torch
from torch import nn, optim
import data
from utils import spatial_softmax

class PoseRegressor(nn.Module):
    # https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf

    def __init__(self, in_channels=3, k=1, num_features=256):
        super(PoseRegressor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, num_features, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(
            num_features, num_features, padding=1, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(
            in_channels=num_features,
            out_channels=k * 2,
            kernel_size=(3, 3),
            stride=1,
            padding=1)

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, C, H, K) tensor.
        """

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


class Generator(nn.Module):
    def __init__(self, in_channels, k_channels, num_features=256):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels + k_channels, num_features, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(
            num_features, num_features, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=num_features,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

def renormalize(heatmaps):
    """
    Args
    ====
    heatmaps: (N, K * 2, H, W)
    
    Returns
    =======
    renormalized_heatmaps (N, K, H, W, 2)
    """
    splits = torch.chunk(heatmaps, 2, dim=1)  # row col
    heatmaps = torch.stack(splits, dim=-1)
    return heatmaps


def f(features):
    row_map = features[..., 0]  # N, K, H, W
    col_map = features[..., 1]  # N, K, H, W
    S_row = row_map.sum(-1)  # N, K, H
    S_col = col_map.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1))).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1))).sum(-1)
    return u_row, u_col


def gaussian_map(features, std=0.2):
    mu = torch.stack(f(features), -1)  # N, K, 2
    mu = mu.unsqueeze(-2).unsqueeze(-2)
    # print(mu.shape)

    dist = torch.distributions.Normal(
        loc=mu,
        scale=torch.ones_like(mu, dtype=mu.dtype) * std,
        # validate_args=True
    )
    height = features.size(2)
    width = features.size(3)

    x, y = torch.meshgrid(
        torch.linspace(-1, 1, width, dtype=mu.dtype),
        torch.linspace(-1, 1, height, dtype=mu.dtype))

    u = torch.stack([x, y], -1).unsqueeze(0).unsqueeze(0)
    return dist.log_prob(u).sum(-1).exp()


class Imm(nn.Module):
    def __init__(self, point_net, generator, std=0.2):
        super(Imm, self).__init__()
        self.point_net = point_net
        self.generator = generator
        self.std = std

    def forward(self, source_images, target_images):

        phi = self.point_net(target_images)

        features = renormalize(spatial_softmax(phi))
        y = gaussian_map(features, std=self.std)
        return self.generator(source_images, y)

def main():
    batch_size = 32
    image_channels = 1
    k = 1

    num_features = 32

    pose_regressor = PoseRegressor(image_channels, k, num_features)
    generator = Generator(image_channels, k, num_features)

    model = Imm(pose_regressor, generator)

    optimizer = torch.optim.Adam(
        model.parameters()
    )

    num_iterations = 1e5
    for it in range(int(num_iterations)):
        xt, xtp1 = data.generate(batch_size)
        xt = torch.as_tensor(xt).unsqueeze(1)
        xtp1 = torch.as_tensor(xtp1).unsqueeze(1)

        optimizer.zero_grad()
        generated = model(xt, xtp1)
        loss = torch.nn.functional.binary_cross_entropy(generated, xtp1)
        
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            loss_mse = torch.nn.functional.mse_loss(generated, xtp1).detach()
            print(it, loss.item(), loss_mse.item())

if __name__ == '__main__':
    main()