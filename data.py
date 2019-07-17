import numpy as np
import matplotlib.pyplot as plt


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