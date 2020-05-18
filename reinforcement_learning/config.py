import torch

env_name = "CartPole-v0"
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 100
goal_score = 200
log_interval = 1
update_target = 10
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLLOW_LANDMAKR = 0
N_DISCRETE_ACTIONS = 5
N_LANDMARKS = 16
WIDTH, HEIGHT, N_CHANNELS = (20, 20, 1)
ACTIONS = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "DONE"}
