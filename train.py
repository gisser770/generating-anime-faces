"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn


from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types_ import Scores, Metrics
from utils.train_utils import TrainParams, my_GaussianNLLLoss, my_KL_divergence
from utils.train_logger import TrainLogger
from generator import evaluate_to_vis, load_my_model
import numpy as np


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0
    epoch_number = 0

    ###### load used model
    load_model = False
    if load_model:
        path = '/home/student/Project/ml_project_vm/logs/VAE_vgg_9_11_20_3_35/model.pth'
        model, epoch_number = load_my_model(model, path)
    ###### end load used model

    # Create optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=train_params.lr_step_size,
    #                                             gamma=train_params.lr_gamma)

    loss_log = np.array([0])
    GaussianNLLLoss_var = torch.mul(torch.ones(1, device='cuda'), train_params.GaussianNLLLoss_var)
    GaussianNLLLoss_var_square = 2 * torch.square(GaussianNLLLoss_var)
    GaussianNLLLoss_var_log = torch.log(GaussianNLLLoss_var)

    for epoch in tqdm(range(epoch_number + 1, train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        gaussian_avg = 0
        kl_avg = 0
        idx = 0
        for i, x in enumerate(train_loader):
            # epoch counter
            idx += 1

            batch_size_ = x.shape[0]

            # forwards
            y_hat, mu, logvar = model(x)

            # GaussianNLLLoss
            gaussian = torch.squeeze(
                my_GaussianNLLLoss(batch_size_, x, y_hat, GaussianNLLLoss_var_square, GaussianNLLLoss_var_log), 0)

            # KL divergence loss
            kl_ = my_KL_divergence(mu, logvar)

            # loss calculation
            loss = kl_ + gaussian

            # set gradients to zero
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # backwards
            optimizer.step()

            # summing loss in order to display avg between epochs
            gaussian_avg += gaussian
            kl_avg += kl_

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            # batch_score = train_utils.compute_score_with_logits(y_hat, y.data).sum()
            # batch_score = train_utils.compute_score_with_logits(y_hat, y).sum()
            # metrics['train_score'] += batch_score.item()
            metrics['train_score'] += gaussian_avg

            metrics['train_loss'] += loss.item()

            # Report model to tensorboard
            if epoch == 0 and i == 0:
                logger.report_graph(model, x)
        print('\n ################################################')
        print(f'gaussian = {gaussian_avg/idx}, kl= {kl_avg/idx}')
        # print(f'forwards time = {forwards_time}, backwards time = {backward_time}, optimizer time = {optimizer_time},'
        #       f' loss time = {loss_time}, batch time = {get_batch_time}')
        print('###############################################')

        # Learning rate scheduler step
        # scheduler.step()
        # TODO: save sample images every ~20 epochs
        # Calculate metrics
        metrics['train_loss'] /= len(train_loader)

        metrics['train_score'] /= len(train_loader)
        # metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        # vis = Visualizations()
        evaluate_to_vis(model, eval_loader, str(epoch), 1)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, train_params)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)
        if epoch == epoch_number + 1:
            best_eval_score = metrics['eval_score'] + 1
        if metrics['eval_score'] < best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, train_params: TrainParams) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :param train_params:
    :return: tuple of (accuracy, loss) values
    """
    GaussianNLLLoss_var = torch.mul(torch.ones(1, device='cuda'), train_params.GaussianNLLLoss_var)
    GaussianNLLLoss_var_square = 2 * torch.square(GaussianNLLLoss_var)
    GaussianNLLLoss_var_log = torch.log(GaussianNLLLoss_var)

    score = 0
    loss = 0
    idx = 0
    for i, x in tqdm(enumerate(dataloader)):
        # if torch.cuda.is_available():
        #     x = x.cuda()
        #     # y = y.cuda()

        y_hat, mu, logvar = model(x)
        batch_size_ = x.shape[0]
        # loss += nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        # score += train_utils.compute_score_with_logits(y_hat, y).sum().item()
        gaussian_log_liklihood = torch.squeeze(
                my_GaussianNLLLoss(batch_size_, x, y_hat, GaussianNLLLoss_var_square, GaussianNLLLoss_var_log), 0)
        kl = my_KL_divergence(mu, logvar)
        loss += gaussian_log_liklihood + kl
        idx += 1
    # loss /= len(dataloader.dataset)
    # score /= len(dataloader.dataset)
    loss /= idx
    score = loss

    return score, loss



