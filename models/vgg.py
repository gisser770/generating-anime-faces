from torch import nn
import torch


def init_normal(m):
	if type(m) == nn.Conv2d:
		nn.init.normal_(m.weight, mean=0, std=0.02)
	if type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
		nn.init.normal_(m.weight, mean=1., std=0.02)

class View(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.view([-1, 512, 2, 2])


def get_vgg_layer(encoder: bool, input_dim: int, output_dim: int, kernel_size: int = None, padding: int = None,
					batch_norm2d: int = None, maxpool2d=None, is_fc_layer: bool = False, fc_need_flatten: bool = False):

	if is_fc_layer:
		if fc_need_flatten:
			if encoder:
				return nn.Sequential(
					nn.Flatten(),
					nn.Linear(input_dim, output_dim)
					)
			else:
				return nn.Sequential(
					nn.Linear(input_dim, output_dim),
					View()
				)
		return nn.Linear(input_dim, output_dim)

	if maxpool2d is not None and maxpool2d != 0:
		if encoder:
			return nn.Sequential(
				nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
				nn.BatchNorm2d(batch_norm2d),
				nn.ReLU(),
				nn.MaxPool2d(maxpool2d)
				)
		else:
			return nn.Sequential(
				nn.UpsamplingBilinear2d(scale_factor=2),
				nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
				nn.BatchNorm2d(batch_norm2d),
				nn.ReLU(),
			)
	else:
		return nn.Sequential(
			nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
			nn.BatchNorm2d(batch_norm2d),
			nn.ReLU()
			)


def create_vgg_layers(latent_dim: int = 512, input_size: list = None, maxpool2d_input: list = None, fc_amount: int = 3):  # for vgg16

	if maxpool2d_input is None:
		maxpool2d_input = [0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0]  # 0 = no maxpool

	if input_size is None:
		input_size = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 2 * 2 * 512, 2048, 1024]

	output_size = list(input_size)
	del output_size[0]
	last_conv_output = output_size[-fc_amount-1]
	output_size[-fc_amount] = last_conv_output
	output_size.append(latent_dim*2)

	# # list of fully connected layers
	is_fc_layer = [False] * (len(input_size) - fc_amount)
	for i in range(fc_amount):
		is_fc_layer.append(True)

	# list of layers that needs nn.Flatten at start
	flatten = [False] * (len(input_size))
	flatten[-fc_amount] = True

	encoder_ = nn.ModuleDict({'en_layer_'+str(i): get_vgg_layer(True, input_size[i], output_size[i], 3, 1,
								output_size[i], maxpool2d_input[i], is_fc_layer[i], flatten[i]) for i in range(16)})
	input_size.reverse()
	is_fc_layer.reverse()
	flatten.reverse()
	output_size.reverse()
	maxpool2d_input.reverse()

	output_size_ = input_size
	is_fc_layer = is_fc_layer
	reshape = flatten
	input_size_ = output_size
	input_size_[0] = latent_dim
	upsample_input = maxpool2d_input
	decoder_ = nn.ModuleDict({'de_layer_'+str(i): get_vgg_layer(False, input_size_[i], output_size_[i], 3, 1,
								output_size_[i], upsample_input[i], is_fc_layer[i], reshape[i]) for i in range(16)})
	return encoder_, decoder_


class VAE_vgg(nn.Module):
	def __init__(self, latent_dim: int = 512, batch_size=64):
		super(VAE_vgg, self).__init__()
		self.z_dim = latent_dim
		self.batch_size = batch_size

		# Encoder + Decoder
		self.encoder_layers, self.decoder_layers = create_vgg_layers(latent_dim=latent_dim)

		# init weights
	def encoder(self, x):
		self.batch_size = x.shape[0]
		for layer_name, layer in self.encoder_layers.items():
			x = layer(x)
		mu = x[:, :self.z_dim]
		logvar = x[:, self.z_dim:]
		return mu, logvar

	def reparameterize(self, mu, logvar):
		stddev = torch.sqrt(torch.exp(logvar))
		epsilon = torch.randn(size=[self.batch_size, self.z_dim], device='cuda')
		return mu + stddev * epsilon

	def decoder(self, x):
		for layer_name, layer in self.decoder_layers.items():
			x = layer(x)
		return x

	def forward(self, x):
		mu, logvar = self.encoder(x)
		x = self.reparameterize(mu, logvar)
		x = self.decoder(x)
		x = x.view(self.batch_size, -1)
		return x, mu, logvar


