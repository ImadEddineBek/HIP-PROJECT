import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforcement_learning.config import gamma
from reinforcement_learning.config import device


# class DQN(nn.Module):
#
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#
#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
#
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))


class QNet(nn.Module):
    def __init__(self, w, h, num_outputs, n_classes=16):
        super(QNet, self).__init__()
        self.num_outputs = num_outputs
        self.n_classes = n_classes
        self.w = w
        self.h = h

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 32
        # print(linear_input_size)

        self.point_detectors = []
        for i in range(n_classes):
            self.point_detectors.append(nn.Sequential(
                nn.ReLU(),
                nn.Linear(linear_input_size, linear_input_size // 4),
                nn.ReLU(),
                nn.Linear(linear_input_size // 4, num_outputs),
            ))
        self.point_detectors = nn.ModuleList(self.point_detectors)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        # self.head = nn.Linear(linear_input_size, num_outputs)
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # print("x", x.size())
        B, L, H, W = x.size()
        # X batch-size, number of landmakrs, 1, H, W
        x = F.relu(self.bn1(self.conv1(x.view(B * L, 1, H, W))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # print("x", x.size())
        B, L, C, H, W = x.size()

        detected_points = torch.zeros([B, L, self.num_outputs], dtype=torch.float32)
        if torch.cuda.is_available():
            detected_points = detected_points.cuda()
        for i in range(L):
            detected_points[:, i] = self.point_detectors[i](x.view(B, L, -1)[:, i])

        # print(B, L, H, W)
        # x_encoded = self.features(x.view(B * L, 1, H, W).float())
        return detected_points

    # TODO
    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        # print(device,batch.state.size())
        states = torch.stack(list(batch.state))
        states = states.view(-1, 1, online_net.h, online_net.w).to(device)
        next_states = torch.stack(batch.next_state).view(-1, 1, online_net.h, online_net.w).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input, i):
        qvalue = self.forward(input)[:, i, :]
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]
