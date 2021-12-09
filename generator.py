from datetime import datetime
from models.base_model import *
# from torchvision.utils import save_image
# import torchvision.transforms as transforms
import dataset
import torchvision
# import visdom
import hydra
from utils import main_utils, train_utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split


def reconstruct(model, epoch, loader):
    image_size = 64
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(loader):
            # if x.shape[0] != batch_size: continue
            batch_size = x.shape[0]
            # data = x.to(device)
            generated, latent_mean, latent_std = model(x)
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], generated.view(batch_size, 3, image_size, image_size)[:n]])
            # comparison = ((comparison + 1) / 2) * 255
            dataset.save_image(comparison.cpu(), './results/reconstruction_' + str(epoch) + '.png', nrow=n)
            return comparison.cpu(), batch_size


def evaluate_to_vis(model, test_loader, epoch, vis):
    # latent_dim = model.module.z_dim

    reconstruction, batch_size = reconstruct(model, epoch, test_loader)
    latent_dim = model.module.z_dim

    index = 'epoch_' + str(epoch) + '.' + str(datetime.now().strftime("%d-%m %Hh%M"))

    with torch.no_grad():
        # sample = torch.randn(batch_size, latent_dim, device='cuda')
        sample = torch.randn(8, latent_dim, device='cuda')
        sample = model.module.decoder(sample).cpu()
        # sample = ((sample + 1) / 2) * 255
        torchvision.utils.save_image(sample.view(-1, 3, 64, 64), './results/random_sample_' + str(index) + '.png')

    # vis.plot_samples(sample, reconstruction, index)
    model.train()


# class Visualizations:
#     def __init__(self, env_name=None):
#         if env_name is None:
#             env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
#         self.env_name = env_name
#         self.vis = visdom.Visdom(env=self.env_name)
#         self.loss_win = None
#
#     def plot_loss(self, testloss, trainloss, epoch):
#         self.loss_win = self.vis.line(
#             [[testloss, trainloss]],
#             [epoch],
#             win=self.loss_win,
#             update='append' if self.loss_win else None,
#             opts=dict(
#                 xlabel='epoch',
#                 ylabel='Loss',
#                 title='Loss',
#                 legend=['Test', 'Train']
#             )
#         )
#
#     def plot_samples(self, sample, reconstruction, index):
#         images = torch.cat([sample.view(-1, 3, 64, 64)[:16], reconstruction[:4], reconstruction[8:12],
#                             reconstruction[4:8], reconstruction[12:16]])
#         images = ((images + 1) / 2) * 255
#         self.vis.images(images, nrow=4, opts=dict(
#             title='Random Generated Samples ' + index,
#         ))
#
#     def plot_reconstruction(self, sample, epoch):
#         self.vis.images(sample, nrow=8, opts=dict(
#             title='Epoch ' + str(epoch) + '(Generated + Reconstructed)',
#         ))


def create_data_loaders(cfg):
    # Load dataset
    data_set = dataset.MyDataset(cfg['main']['paths']['train']['path'],
                                 cfg['main']['paths']['train']['number_of_samples'])
    dataset_len = len(data_set)
    train_size = int(cfg['main']['train_size'] * dataset_len)
    val_size = dataset_len - train_size
    train_dataset, val_dataset = random_split(data_set, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(cfg['main']['seed']))

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    return train_loader, eval_loader


def create_my_data_loaders(hdf5_path, num_images):
    # Load dataset
    data_set = dataset.MyDataset(hdf5_path, num_images)
    dataset_len = len(data_set)

    eval_loader = DataLoader(data_set, 5, shuffle=True, num_workers=0)
    return eval_loader


def load_my_model(model, path, lr=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch_number = checkpoint['epoch']
    # return model, optimizer, epoch_number
    return model, epoch_number


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    main_utils.init(cfg)
    train_loader, eval_loader = create_data_loaders(cfg)

    model = VAE()
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    train_params = train_utils.get_train_params(cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=train_params.lr)

    checkpoint = torch.load('/home/student/Project/VAE_9_3_23_47_8_-_weighted_loss_50_100/model.pth')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    evaluate_to_vis(model, eval_loader, 'epoch', 1)


if __name__ == '__main__':
    main()


