import argparse
import os
from comet_ml import Experiment

# from data_loader import get_loader
from trainers.trainer2D import Trainer2D
from torch.backends import cudnn

from utils.utils import fix_path


def main(config):
    # svhn_loader, mnist_loader, mnist_val_loader = get_loader(config)
    # print("loaded")
    solver = Trainer2D(config)
    # cudnn.benchmark = True

    # create directories if not exist
    # if not os.path.exists(config.model_path):
    #     os.makedirs(config.model_path)
    '''if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)'''
    solver.pre_train()
    solver.train()


if __name__ == '__main__':
    fix_path()
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=350)
    # parser.add_argument('--alpha_s', type=float, default=0.5)
    # parser.add_argument('--alpha_t', type=float, default=0.8)
    # parser.add_argument('--beta_c', type=float, default=1)
    # parser.add_argument('--beta_sep', type=float, default=1.5)
    # parser.add_argument('--beta_p', type=float, default=4)

    # training hyper-parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_c', type=float, default=0.003)

    # misc
    parser.add_argument('--mode', type=str, default='2D')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_csv_train', type=str, default='./data/data.csv')
    parser.add_argument('--data_csv_test', type=str, default='./data/data.csv')
    parser.add_argument('--data_csv_path', type=str, default='./data/data.csv')
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)
