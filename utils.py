import numpy as np
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


def _maybe_convert_dict(value):
    if isinstance(value, dict):
        return ConfigDict(value)

    return value


class ConfigDict(dict):
    """Configuration container class."""

    def __init__(self, initial_dictionary=None):
        """Creates an instance of ConfigDict.
        Args:
            initial_dictionary: Optional dictionary or ConfigDict containing initial
            parameters.
        """
        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _maybe_convert_dict(value)
        super(ConfigDict, self).__init__(initial_dictionary)

    def __setattr__(self, attribute, value):
        self[attribute] = _maybe_convert_dict(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor)
            for x in [np.random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for i in range(n):
        colors.append(generate_new_color(colors, pastel_factor=0.9))
    return colors
