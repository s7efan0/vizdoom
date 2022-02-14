import os
import time
import imageio
import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from vizdoom import *


# class CustomCNN(BaseFeaturesExtractor):
#
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(4, 4), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(2, 2), padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]
#
#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))


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
        self.game.load_config("../ViZDoom-master/scenarios/take_cover.cfg")
        self.game.set_window_visible(render)
        self.game.set_render_hud(False)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(2)

    def step(self, action):
        actions = np.identity(2, dtype=np.uint8)
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
    env.reset()
    print(env.game.get_state().screen_buffer.shape)

    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch=[32, 32, dict(vf=[32, 32], pi=[16, 8])]
    # )

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
    print(model.policy)
    model.learn(100000, callback=callback)

    env.close()


def test():
    model = PPO.load("./train/model100000.zip")
    env = VizDoomEnv(True)

    print(model.policy)

    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action[0])
            time.sleep(0.02)
            total_reward += reward
        print(f"Total reward for episode {episode + 1} is {total_reward}")
        time.sleep(1)

    env.close()


def makeGif():
    model = PPO.load("./train/model100000.zip")
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

    imageio.mimsave('takeCover.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=15)


if __name__ == '__main__':
    makeGif()
