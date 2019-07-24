import numpy as np
import torch
import json
from PIL import Image
import torch.utils.data


def generate(num_samples=1, 
             size=16, dx=5):
    im = np.zeros((num_samples, size, size), dtype=np.float32)
    imt = np.zeros((num_samples, size, size), dtype=np.float32)
    
    x = np.random.randint(size, size=(num_samples,))
    y = np.random.randint(size, size=(num_samples,))
    
    for di in range(-1, 2):
        for dj in range(-1, 2):
            im[np.arange(num_samples), 
               np.clip(y + di, 0, size-1), 
               np.clip(x + dj, 0, size-1)] = 1.
    # im[:, y][x] = 1.
    
    
    # dx = np.random.randint(0, 5, size=(num_samples))
    dx = np.ones((num_samples, ), dtype=np.int) * dx
    
    for di in range(-1, 2):
        for dj in range(-1, 2):
            imt[np.arange(num_samples),
                (np.clip(y + di, 0, size-1)), 
                (np.clip(x + dx + dj, 0, size-1))] = 1.
    
    return im, imt
    
def vis_sample(sample):
    im, imt = sample
    im = np.concatenate(im, 0)
    imt = np.concatenate(imt, 0)
    print(im.shape)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im, cmap='gray')
    ax[1].imshow(imt, cmap='gray')
    for a in ax.flat:
        a.set_axis_off()


class Dataset(object):
    _meta_data_file = 'metadata.json'
    def __init__(self, root, transform=None):
        self._root = root
        self._transform = transform
        with open('{}/{}'.format(root, self._meta_data_file), 'rt') as inp:
            self._metadata = json.load(inp)

    @property
    def num_trajectories(self):
        return self._metadata['num_trajectories']

    @property
    def num_timesteps(self):
        return self._metadata['num_timesteps']

    def __len__(self):
        raise NotImplementedError

    def get_image(self, n, t):
        im = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, t)))
        return im

    def __getitem__(self, idx):
        n, t, tp1 = idx
        imt = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, t)))
        imtp1 = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, tp1)))
        if self._transform is not None:
            imt = self._transform(imt)
            imtp1 = self._transform(imtp1)

        return imt, imtp1
    
    def get_trajectory(self, idx):
        images = [np.array(Image.open('{}/{}/{}.png'.format(self._root, idx, t))) for t in range(self.num_timesteps)]
        return [self._transform(im) for im in images]

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        while True:
            n = np.random.randint(self._dataset.num_trajectories)
            num_images = self._dataset.num_timesteps
            t_ind = np.random.randint(0, num_images - 20)
            tp1_ind = t_ind + np.random.randint(20)
            yield n, t_ind, tp1_ind

    def __len__(self):
        raise NotImplementedError
