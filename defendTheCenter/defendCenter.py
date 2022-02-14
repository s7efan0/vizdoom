import imageio
from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box
import time
import numpy as np
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'model{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def grayscale(observation):
    grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(grey, (160, 100), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100, 160, 1))
    return state


class VizDoomEnv(Env):
    def __init__(self, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config("../ViZDoom-master/scenarios/defend_the_center.cfg")
        self.game.set_window_visible(render)
        self.game.set_render_hud(False)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = grayscale(state)
            ammo = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            ammo = 0

        info = {"ammo": ammo}
        done = self.game.is_episode_finished()

        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return grayscale(state)

    def close(self):
        self.game.close()


def train():
    CHECKPOINT_DIR = './train'
    LOG_DIR = './logs'
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    env = VizDoomEnv(False)

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99)
    model.learn(70000, callback=callback)

    env.close()


def test():
    model = PPO.load("./train/model70000.zip")
    env = VizDoomEnv(True)

    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action[0])
            time.sleep(0.01)
            total_reward += reward
        print(f"Total reward for episode {episode + 1} is {total_reward}")
        time.sleep(1)

    env.close()


def makeGif():
    model = PPO.load("./train/model70000.zip")
    env = VizDoomEnv(True)
    images = []

    for episode in range(5):
        state = env.reset()
        done = False

        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action[0])
            if env.game.get_state():
                images.append(np.moveaxis(env.game.get_state().screen_buffer, 0, -1))
            else:
                images.append(np.zeros((240, 320, 3)))

    imageio.mimsave('defendCenter.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)


if __name__ == '__main__':
    makeGif()
