import torch
import sys

from comet_ml import Experiment
import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage
import torch.nn as nn
from termcolor import colored
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataloaders.dataloader2D import get_dataloader2D, get_dataloader2DJigSaw
from models import conv2d
from reinforcement_learning.landmark_detection_envirenment import LandmarkEnv
from utils.evaluate import Evaluator
from reinforcement_learning.config import env_name, initial_exploration, batch_size, update_target, goal_score, \
    log_interval, device, \
    replay_memory_capacity, lr, N_LANDMARKS, HEIGHT, WIDTH
import torch
import torch.optim as optim
import torch.nn.functional as F
from reinforcement_learning.model import QNet
from reinforcement_learning.memory import Memory


def get_action(state, target_net, epsilon, env):
    state = state.view(-1, 1, WIDTH, HEIGHT).to(device)
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


class TrainerRL:
    def __init__(self, config):
        self.experiment = Experiment(api_key='CQ4yEzhJorcxul2hHE5gxVNGu', project_name='HIP')
        self.experiment.log_parameters(vars(config))
        self.config = config
        self.train_loader, self.test_loader = get_dataloader2D(config)

        self.env = LandmarkEnv(self.train_loader)
        self.num_inputs = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        print('state size:', self.num_inputs)
        print('action size:', self.num_actions)
        self.online_net = []
        self.target_net = []
        self.optimizer = []
        self.memory = []
        for i in range(N_LANDMARKS):
            self.online_net.append(QNet(self.num_inputs[0], self.num_inputs[1], self.num_actions))
            self.target_net.append(QNet(self.num_inputs[0], self.num_inputs[1], self.num_actions))
            update_target_model(self.online_net[i], self.target_net[i])

            self.optimizer.append(optim.Adam(self.online_net[i].parameters(), lr=lr))

            self.online_net[i].to(device)
            self.target_net[i].to(device)
            self.online_net[i].train()
            self.target_net[i].train()
            self.memory.append(Memory(replay_memory_capacity))

        self.running_score = 0
        self.epsilon = 1.0
        self.steps = 0
        self.loss = 0

        # self.log_step = config.log_step
        # self.model = conv2d.Conv2DPatches(image_size=config.image_size)
        # print(self.model)
        # self.d = get_dataloader2D(config)
        # self.train_loader, self.test_loader = self.d
        # self.train_loader_jig, self.test_loader_jig = get_dataloader2DJigSaw(config)
        # self.net_optimizer = optim.Adam(self.model.parameters(), config.lr, [0.5, 0.9999])
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        # self.criterion_c = nn.CrossEntropyLoss()
        # self.criterion_d = nn.MSELoss()
        # self.epochs = config.epochs
        # if torch.cuda.is_available():
        #     print('Using CUDA')
        #     self.model = self.model.cuda()
        # #     self.model = self.model.cuda()
        # self.pre_model_path = './artifacts/pre_models/' + str(config.lr) + '.pth'
        # self.model_path = './artifacts/models/' + str(config.lr) + '.pth'
        # self.image_size = config.image_size

    def train(self):
        # if os.path.isfile(self.model_path):
        #     print("Using pre-trained model")
        #     self.model = torch.load(self.model_path)
        if False:
            pass
        else:
            randomness = 1
            for e in range(3000):
                done = False
                running_score = 0
                epsilon = 1.0
                steps = 0
                loss = 0
                score = 0
                state = self.env.reset(randomness)
                # print(state.shape)
                state = torch.Tensor(state).to(device)
                # state = state.unsqueeze(0)
                while not done and steps < 500:
                    # print(steps)
                    steps += 1
                    actions = []
                    for i in range(N_LANDMARKS):
                        actions.append(get_action(state[i], self.target_net[i], epsilon, self.env))

                    next_state, reward, done, _ = self.env.step(actions)
                    # print(next_state.shape)

                    next_state = torch.Tensor(next_state)
                    # next_state = next_state.unsqueeze(0)

                    reward = reward  # if not done or score == 499 else -1
                    action_one_hot = np.zeros((N_LANDMARKS, self.num_actions))
                    for i in range(N_LANDMARKS):
                        mask = 0 if done[i] else 1
                        action_one_hot[i][actions[i]] = 1
                        self.memory[i].push(state[i].to(device), next_state[i].to(device), action_one_hot[i], reward[i],
                                            mask)

                    score += np.mean(reward)
                    state = next_state
                    done = done.all()
                    if steps > initial_exploration:
                        # print(steps)
                        epsilon -= 0.05
                        if steps % 100:
                            randomness += 1
                        randomness = min(randomness, 50)
                        epsilon = max(epsilon, 0.1)
                        for i in range(N_LANDMARKS):
                            batch = self.memory[i].sample(batch_size)
                            # print(batch.state)
                            loss = QNet.train_model(self.online_net[i], self.target_net[i], self.optimizer[i], batch)

                            if steps % update_target == 0:
                                update_target_model(self.online_net[i], self.target_net[i])

                # score = score if score == 500.0 else score + 1
                # running_score = 0.99 * running_score + 0.01 * score
                if e % log_interval == 0:
                    print('{} episode | steps: {:.2f} | epsilon: {:.2f} | distance: {:.2f} | done {:.2f}'.format(
                        e, steps, epsilon, np.mean(self.env.distances), np.mean(done)))
                    # play(env, target_net, epsilon)
                # if running_score > goal_score:
                #     break
            for epoch in range(1):
                self.env.reset()
                for _ in range(30):
                    self.env.render()
                    obs, reward, done, info = self.env.step([self.env.action_space.sample()] * 16)
                    print(done)
                self.env.done()
            self.env.close()
        #     print("Starting training")
        #     if torch.cuda.is_available():
        #         self.model = self.model.cuda()
        #     for epoch in range(self.epochs):
        #         print("Starting epoch {}".format(epoch))
        #         train_loader = iter(self.train_loader)
        #         with self.experiment.train():
        #             for i in range(len(train_loader)):
        #                 self.net_optimizer.zero_grad()
        #                 data, landmarks, _ = train_loader.next()
        #                 # print(landmarks)
        #                 data, landmarks = self.to_var(data), self.to_var(landmarks)
        #                 B, L, H, W = data.size()
        #                 B, L, S = landmarks.size()
        #                 y = landmarks[:, :, 1].view(B, L)
        #                 y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
        #                 if torch.cuda.is_available():
        #                     y_slices = y_slices.cuda()
        #                 for i in range(B):
        #                     y_slices[i] = data[i, y[i]]
        #
        #                 jig_out, detected_points = self.model(y_slices)
        #                 landmarks = landmarks.float() / self.image_size
        #                 loss = self.criterion_d(detected_points, landmarks[:, :, [0, 2]])
        #                 loss.backward()
        #                 self.net_optimizer.step()
        #                 # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
        #                 self.experiment.log_metric('loss', loss.item())
        #                 print('loss: {}'.format(loss.item()))
        #         if epoch % self.log_step == 0:
        #             with self.experiment.test():
        #                 self.evaluate()
        #                 evaluator = Evaluator(self, self.test_loader)
        #                 evaluator.report()
        #     torch.save(self.model, self.model_path)
        # evaluator = Evaluator(self, self.test_loader)
        # evaluator.report()

    def evaluate(self):
        test_loader = iter(self.test_loader)
        # with self.experiment.test():
        #     loss = 0
        #     for i in range(len(test_loader)):
        #         self.net_optimizer.zero_grad()
        #         data, landmarks, _ = test_loader.next()
        #         data, landmarks = self.to_var(data), self.to_var(landmarks)
        #         B, L, H, W = data.size()
        #         B, L, S = landmarks.size()
        #         y = landmarks[:, :, 1].view(B, L)
        #         y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
        #         if torch.cuda.is_available():
        #             y_slices = y_slices.cuda()
        #
        #         for i in range(B):
        #             y_slices[i] = data[i, y[i]]
        #
        #         jig_out, detected_points = self.model(y_slices)
        #         landmarks = landmarks.float() / self.image_size
        #         loss += self.criterion_d(detected_points, landmarks[:, :, [0, 2]]).item()
        #         self.plots(y_slices.cpu(), landmarks[:, :, [0, 2]], detected_points)
        #     self.experiment.log_metric('loss', loss / len(test_loader))

    def plots(self, slices, real, predicted):
        # figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        # slices = slices[0].cpu().detach().numpy()
        # real = real[0].cpu().detach().numpy()
        # predicted = predicted[0].cpu().detach().numpy()
        # real *= self.image_size
        # predicted *= self.image_size
        # s = 0
        # # print(real.size())
        # # print(predicted.size())
        # for i in range(4):
        #     for j in range(4):
        #         axes[i, j].imshow(slices[s])
        #         x, z = real[s]
        #         axes[i, j].scatter(x, z, color='red')
        #         x, z = predicted[s]
        #         axes[i, j].scatter(x, z, color='blue')
        #         s += 1
        # self.experiment.log_figure(figure=plt)
        # plt.savefig('artifacts/predictions/img.png')
        # plt.show()
        ...

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=False)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def predict(self, x):
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        #     x = x.cuda()
        # _, x = self.model(x)
        # return x
        ...
