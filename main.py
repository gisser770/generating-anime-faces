
"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train
from dataset import MyDataset
from models.vgg import VAE_vgg
from models.base_model import VAE, init_normal
from torch.utils.data import DataLoader, random_split
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf


torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    torch.backends.cudnn.benchmark = True
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Init model
    model = VAE_vgg(latent_dim=cfg['main']['auto_encoder_latent_dim'])
    # model = VAE(latent_dim=cfg['main']['auto_encoder_latent_dim'])
    model.apply(init_normal)

    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    # Load dataset
    dataset = MyDataset(cfg['main']['paths']['train']['path'], cfg['main']['paths']['train']['number_of_samples'])
    dataset_len = len(dataset)
    train_size = int(cfg['main']['train_size'] * dataset_len)
    val_size = dataset_len - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(cfg['main']['seed']))

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_loader, train_params, logger)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
