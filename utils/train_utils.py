"""
Includes all utils related to training
"""

import torch
import time
from typing import Dict
from torch import Tensor
from omegaconf import DictConfig


def compute_score_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels
    :return: score for each sample
    """
    # logits = torch.max(logits, 1)[1].data  # argmax
    #
    # logits_one_hots = torch.zeros(*labels.size())
    # if torch.cuda.is_available():
    #     logits_one_hots = logits_one_hots.cuda()
    # logits_one_hots.scatter_(1, logits.view(-1, 1), 1)
    #
    # scores = (logits_one_hots * labels)

    # return scores
    return torch.tensor(1.0)


def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train_loss': 0, 'train_score': 0, 'total_norm': 0, 'count_norm': 0}


class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    grad_clip: float
    save_model: bool

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.grad_clip = kwargs['grad_clip']
        self.save_model = kwargs['save_model']
        self.GaussianNLLLoss_var = kwargs['GaussianNLLLoss_var']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])


class TimeMeasure:
    def __init__(self):
        self.start = 0
        self.name = str()

    def start(self, name):
        self.start = time.time()
        self.name = name
        print(f'{name} start')

    def end(self, name):
        end = time.time()
        print(f'{name} end, duration = {end - self.start}(sec)')


def my_GaussianNLLLoss(batch_size, x, y_hat, var_square, var_log):
    # var = torch.unsqueeze(torch.mul(torch.ones(64, device='cuda'), 0.01), 1)
    # batch_size = x.shape[0]
    # var = torch.mul(torch.ones(1, device='cuda'), 0.01)
    # var = torch.mul(torch.ones(1, device='cuda'), 0.1)
    # x_flat = x.view(batch_size, -1)
    # gaussian = (0.5 * torch.sum(torch.square(x_flat - y_hat)) / (2 * torch.square(var)) + torch.log(var)) / batch_size
    gaussian = (0.5 * torch.sum(torch.square(x.view(batch_size, -1) - y_hat)) / var_square + var_log) / batch_size
    return gaussian
    # return gaussian


def my_KL_divergence(mu, logvar):
    batch_size = logvar.shape[0]
    # kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch_size  # logvar.shape[0] = batch size

    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch_size
    # return min(kl, torch.tensor(200000000, device='cuda'))
    return kl

