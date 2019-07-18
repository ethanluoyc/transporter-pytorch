import baselines
import gym
import os
from PIL import Image
import json
import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame
from torchvision import transforms
from tqdm import tqdm


def main():
    num_trajectoris = 10
    datadir = 'data/'
    num_steps = 100

    def make_env(env_id, num_steps):
        env = make_atari(env_id, max_episode_steps=num_steps)
        return WarpFrame(env, 80, 80, grayscale=False)

    env = make_env('PongNoFrameskip-v4', num_steps)
    obs = env.reset()
    with tqdm(total=num_trajectoris * num_steps) as pbar:
        for n in range(num_trajectoris):
            os.makedirs('{}/{}'.format(datadir, n), exist_ok=True)
            obs = env.reset()
            t = 0
            Image.fromarray(obs).save('{}/{}/{}.png'.format(datadir, n, t))
            images = []
            while True:
                obs, r, done, _ = env.step(env.action_space.sample())
                Image.fromarray(obs).save('{}/{}/{}.png'.format(datadir, n, t))
                images.append(obs)
                t += 1
                if done:
                    break
            pbar.update(num_steps)
        with open('{}/metadata.json'.format(datadir), 'w') as out:
            json.dump({
                'num_trajectories': num_trajectoris,
                'num_timesteps': num_steps
            }, out)


if __name__ == '__main__':
    main()
