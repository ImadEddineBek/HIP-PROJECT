import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from reinforcement_learning.config import N_DISCRETE_ACTIONS, N_CHANNELS, WIDTH, HEIGHT, N_LANDMARKS, FOLLOW_LANDMAKR


def calculate_reward(pred, target, action):
    x, y = target[0], target[1]
    x_, y_ = pred[0], pred[1]
    reward = -0.5
    if x < x_:
        if action == 0:
            reward += 1
        elif action == 1:
            reward += -5

    if x > x_:
        if action == 0:
            reward += -5
        elif action == 1:
            reward += 1

    if x == x_:
        if action == 0 or action == 1:
            reward += -5

    if y < y_:
        if action == 2:
            reward += 1
        elif action == 3:
            reward += -5

    if y > y_:
        if action == 2:
            reward += -5
        elif action == 3:
            reward += 1

    if y == y_:
        if action == 2 or action == 3:
            reward += -5

    if y == y_ and x == x_:
        if action == 4:
            reward += 10
    # print(pred, target, action, reward)

    return reward


class LandmarkEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, loader):
        super(LandmarkEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.loader = loader
        self._loader = iter(self.loader)
        self.nb_samples = len(self._loader)
        self.i = 0
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        if N_CHANNELS == 1:
            self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, N_CHANNELS), dtype=np.uint8)

    def step(self, actions):
        self.current_step += 1
        # Execute one time step within the environment
        rewards = []
        dones = []
        self.distances = []
        for i in range(N_LANDMARKS):
            action = actions[i]
            new_x, new_y = self.current[i]
            # print(new_x, new_y)
            if action == 0:
                new_x = self.current[i][0] - 1
            elif action == 1:
                new_x = self.current[i][0] + 1
            elif action == 2:
                new_y = self.current[i][1] - 1
            elif action == 3:
                new_y = self.current[i][1] + 1
            elif action == 4:
                pass

            reward = calculate_reward(self.current[i], self.index[i], action)
            # - self.current_step / 5 - np.linalg.norm(self.current[i] - self.index[i]) / self.H
            self.distances.append(np.linalg.norm(self.current[i] - self.index[i]))
            if new_y < 0:
                new_y = 0
                reward -= 1
            elif new_y >= self.W - 1:
                new_y = self.W - 1
                reward -= 1

            if new_x <= 0:
                new_x = 0
                reward -= 1
            elif new_x >= self.H - 1:
                new_x = self.H - 1
                reward -= 1

            self.current[i] = np.array([new_x, new_y])

            done = (self.index[i] == self.current[i]).all()
            if done:
                reward = 100
            dones.append(done)
            rewards.append(reward)
        return self._next_observation(), np.array(rewards), np.array(dones), {}

    def reset(self, randomness=1):
        # Reset the state of the environment to an initial state re
        self.current_step = 0
        if self.i == self.nb_samples - 1:
            self._loader = iter(self.loader)
            self.i = 0
        else:
            self.i += 1

        data, landmarks, _ = self._loader.next()
        B, L, H, W = data.size()
        B, L, S = landmarks.size()
        data = data[0]
        landmarks = landmarks[0]
        y = landmarks[:, 1].view(L)
        # y_slices = torch.zeros([L, H, W], dtype=torch.float32)
        # if torch.cuda.is_available():
        #     y_slices = y_slices.cuda()
        y_slices = data[y].numpy()

        self.image = y_slices  # np.flip(y_slices, 1)
        # print(self.image.shape)
        L, self.W, self.H = self.image.shape
        if not L == N_LANDMARKS:
            exit()
        self.index = landmarks.cpu().numpy()[:, 0:3:2]
        # print(self.index)
        self.current = self.index + np.random.randint(-randomness, randomness, (L, 2))
        self.frames = []
        return self._next_observation()

    def _next_observation(self):
        """
        return a list of images of the location of current landmarks shape (N_LANDMARKS, WIDTH, HEIGHT).
        """
        result = np.zeros((N_LANDMARKS, WIDTH, HEIGHT))
        for i in range(N_LANDMARKS):
            x_center, y_center = self.current[i][0], self.current[i][1]
            if x_center < WIDTH // 2:
                x_center = WIDTH // 2 + 1
            if x_center >= self.W - (WIDTH // 2):
                x_center = self.W - (WIDTH // 2) - 1

            if y_center < HEIGHT // 2:
                y_center = HEIGHT // 2 + 1
            if y_center >= self.H - (HEIGHT // 2):
                y_center = self.H - (HEIGHT // 2) - 1

            x_start = x_center - (WIDTH // 2)
            x_finish = x_center + (WIDTH // 2)

            y_start = y_center - (HEIGHT // 2)
            y_finish = y_center + (HEIGHT // 2)
            if i == FOLLOW_LANDMAKR:
                self.y_start = y_start
                self.x_start = x_start
            result[i] = self.image[i, x_start:x_finish, y_start:y_finish]
        return result

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        fig, ax = plt.subplots(1)

        ax.imshow(self.image[FOLLOW_LANDMAKR])
        # Create a Rectangle patch
        rect = patches.Rectangle((self.x_start, self.y_start), WIDTH, HEIGHT, linewidth=1, edgecolor="r",
                                 facecolor="none")

        # Add the patch to the Axes
        ax.add_patch(rect)
        x_center, y_center = self.current[FOLLOW_LANDMAKR]
        plt.scatter(x_center, y_center, label="pred")
        x_center, y_center = self.index[FOLLOW_LANDMAKR]
        plt.scatter(x_center, y_center, label="real")
        # plt.ylim(0, self.W - 1)
        # plt.xlim(0, self.H - 1)
        plt.legend()
        plt.savefig(f"artifacts/predictions/img_{self.i}_{self.current_step}.png")
        # self.frames.append(plt.show(block=False).get_array())
        # plt.pause(3)
        # plt.close()

    def done(self):
        import cv2
        import glob

        img_array = []
        for filename in glob.glob(f"artifacts/predictions/img_{self.i}_*.png"):
            import os

            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            # img = cv2.flip(img, 1)
            img_array.append(img)
            os.remove(filename)

        out = cv2.VideoWriter(f"artifacts/predictions/video_{self.i}.avi", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
