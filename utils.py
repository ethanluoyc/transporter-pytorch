import torch.nn.functional as F

def spatial_softmax(features):
    """Compute softmax over the spatial dimensions

    Compute the softmax over heights and width

    Args
    ----
    features: tensor of shape [N, C, H, W]
    """
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output
