import argparse
import os

from data_loader import get_loader
from trainers.trainer2D import Trainer2D
from torch.backends import cudnn

from utils.utils import fix_path


def main(config):
    svhn_loader, mnist_loader, mnist_val_loader = get_loader(config)
    print("loaded")
    solver = Trainer2D(config)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    '''if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)'''
    if config.mode == 'train':
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
    parser.add_argument('--train_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr_c', type=float, default=0.0003)

    # misc
    parser.add_argument('--mode', type=str, default='2D')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_csv_path', type=str, default='./data/data.csv')
    parser.add_argument('--data_img_root_path', type=str, default='')
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)
