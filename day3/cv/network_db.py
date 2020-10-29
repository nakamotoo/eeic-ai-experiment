
import torch
import torch.nn as nn
from torchvision.models import vgg16
from network import MLP

class Vgg16(torch.nn.Module):
	"""
	Network using VGG-16 for feature extraction

	"""
	def __init__(self):
		"""
		Construct copy of VGG-16
		See https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html for details
		"""
		super(Vgg16, self).__init__()
		v = vgg16(pretrained= True)
		# Copy modules of vgg16
		features = list(v.features)
		avgpool = v.avgpool
		sequentials = list(v.classifier)
		self.features = nn.ModuleList(features).eval() 
		self.avgpool = avgpool
		self.sequentials = nn.ModuleList(sequentials).eval() 
            
	def forward(self, x):
		"""
		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		x	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		for i,model in enumerate(self.features):
			x = model(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		for i,model in enumerate(self.sequentials):
			x = model(x)
			# Return output of 1st layer in sequentials
# 			if i == 3:
			if i == 0:
				return x

# 課題4.3.4
class MLP_DB(torch.nn.Module):

	def __init__(self, model_path):
		super(MLP_DB, self).__init__()
		v = MLP(5000, 28*28, 10)
		v.load_state_dict(torch.load( model_path))
		print(v)

		self.fc1 = v.fc1
		self.fc2 = v.fc2
		self.fc3 = v.fc3
            
	def forward(self, x):
		print(x.shape)
		x = self.fc1(x)
		x = self.fc2(x)
		print(x.shape)
		return x