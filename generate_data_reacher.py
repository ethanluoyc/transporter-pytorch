import os
import json
import argparse

from PIL import Image
import baselines
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import gym
import pybullet
from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv


def main():
    parser = argparse.ArgumentParser(description='Generate Reacher trajectories.')
    parser.add_argument('--datadir', default='data')
    parser.add_argument('--num_steps', default=100, type=int)
    parser.add_argument('--num_trajectories', default=10, type=int)
    parser.add_argument('--seed', default=4242, type=int)
    args = parser.parse_args()

    num_trajectories = args.num_trajectories
    datadir = args.datadir
    num_steps = args.num_steps
    np.random.seed(args.seed)
    

    pybullet.connect(pybullet.DIRECT)

    class Env(ReacherBulletEnv):
        def __init__(self):
            super(Env, self).__init__(render=True)
            self._cam_dist = .5
            self._cam_yaw = 0
            self._cam_pitch = 90
            self._render_width = 320
            self._render_height = 320

    class SkipFrame(gym.Wrapper):
        def step(self, action):
            for _ in range(4):
                obs, r, done, info = self.env.step(action)
            return obs, r, done, info

    env = SkipFrame(Env())

    print("Data will be saved to {}".format(datadir))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((80, 80)),
    ])


    with tqdm(total=num_trajectories * num_steps) as pbar:
        for n in range(num_trajectories):
            os.makedirs('{}/{}'.format(datadir, n), exist_ok=True)
            obs = env.reset()
            obs = env.render('rgb_array')
            t = 0
            transform(obs).save('{}/{}/{}.png'.format(datadir, n, t))
            images = []
            while t < num_steps:
                obs, r, done, _ = env.step(env.action_space.sample())
                obs = env.render('rgb_array')
                transform(obs).save('{}/{}/{}.png'.format(datadir, n, t))
                t += 1
                if done:
                    break
            pbar.update(num_steps)
        with open('{}/metadata.json'.format(datadir), 'w') as out:
            json.dump({
                'num_trajectories': num_trajectories,
                'num_timesteps': num_steps
            }, out)


if __name__ == '__main__':
    main()
