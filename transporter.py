import torch
from torch import nn
from utils import spatial_softmax


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1,
                 padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class FeatureEncoder(nn.Module):
    """Phi"""

    def __init__(self, in_channels=3):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            Block(in_channels, 32, kernel_size=(7, 7), stride=1, padding=3), # 1
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 128, kernel_size=(3, 3), stride=2), # 5
            Block(128, 128, kernel_size=(3, 3), stride=1),  # 6        
        )


    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x)


class PoseRegressor(nn.Module):
    """Pose regressor"""

    # https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf

    def __init__(self, in_channels=3, k=1):
        super(PoseRegressor, self).__init__()
        self.net = nn.Sequential(
            Block(in_channels, 32, kernel_size=(7, 7), stride=1, padding=3), # 1
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 128, kernel_size=(3, 3), stride=2), # 5
            Block(128, 128, kernel_size=(3, 3), stride=1),  # 6        
        )
        self.regressor = nn.Conv2d(128, k, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.
        
        Returns
        =======
        y: (N, k, H', W') tensor.
        """
        x = self.net(x)
        return self.regressor(x)


class RefineNet(nn.Module):
    """Network that generates images from feature maps and heatmaps."""

    def __init__(self, num_channels):
        super(RefineNet, self).__init__()
        self.net = nn.Sequential(
            Block(128, 128, kernel_size=(3, 3), stride=1), # 6 
            Block(128, 64, kernel_size=(3, 3), stride=1), # 5
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, num_channels, kernel_size=(7, 7), stride=1, padding=3), # 1
        )


    def forward(self, x):
        """
        x: the transported feature map.
        """
        return self.net(x)


def renormalize(heatmaps):
    """
    Args
    ====
    heatmaps: (N, K * 2, H, W)
    
    Returns
    =======
    renormalized_heatmaps (N, K, H, W, 2)
    """

    return heatmaps


def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1) # N, K, 2


def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu = mu.unsqueeze(-2).unsqueeze(-2) # N, K, 1, 1, 2

    dist = torch.distributions.Normal(
        loc=mu,
        scale=torch.ones_like(mu, dtype=mu.dtype, device=features.device) * std,
    )  # N, K, 1, 1, 2
    height = features.size(-2)
    width = features.size(-1)

    x, y = torch.meshgrid(
        torch.linspace(-1, 1, width, dtype=mu.dtype, device=mu.device),
        torch.linspace(-1, 1, height, dtype=mu.dtype, device=mu.device))
    # x, y (H, W)
    u = torch.stack([x, y], -1).unsqueeze(0).unsqueeze(0) # u: (1, 1, H, W, 2)
    # N, K, H, W, 2 -> NKHW
    return dist.log_prob(u).sum(-1).exp()


def transport(source_keypoints, target_keypoints, source_features,
              target_features):
    """
    Args
    ====
    source_keypoints (N, K, H, W)
    target_keypoints (N, K, H, W)
    source_features (N, D, H, W)
    target_features (N, D, H, W)

    Returns
    =======
    """
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * source_features + t.unsqueeze(1) * target_features
    return out


class Transporter(nn.Module):

    def __init__(self, feature_encoder, point_net, refine_net, std=0.1):
        super(Transporter, self).__init__()
        self.feature_encoder = feature_encoder
        self.point_net = point_net
        self.refine_net = refine_net
        self.std = std

    def forward(self, source_images, target_images):
        source_features = self.feature_encoder(source_images)
        target_features = self.feature_encoder(target_images)

        source_keypoints = gaussian_map(
            spatial_softmax(self.point_net(source_images)), std=self.std)

        target_keypoints = gaussian_map(
            spatial_softmax(self.point_net(target_images)), std=self.std)

        transported_features = transport(source_keypoints.detach(),
                                         target_keypoints,
                                         source_features.detach(),
                                         target_features)

        assert transported_features.shape == target_features.shape

        reconstruction = self.refine_net(transported_features)
        return reconstruction
