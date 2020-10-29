
import argparse
import glob
import os
import torch
from torchvision import transforms
from PIL import Image
from network_db import MLP_DB

def createDatabase(paths, gpu, model_path):
	# Create model
	model = MLP_DB(model_path)
	# Set transformation
	data_preprocess = transforms.Compose(	[
		transforms.Grayscale(),
		transforms.Resize((28, 28)),
		transforms.ToTensor()] )

	# Set model to GPU/CPU
	device = 'cpu'
	if gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(gpu)
	model = model.to(device)
	print(model)

# Get features
	with torch.no_grad():
		features = torch.cat(
			[model(data_preprocess(Image.open(path, 'r')).unsqueeze(0).to(device).view(-1, 28*28)).to('cpu')
				for path in paths],
			dim = 0
		)
	# Show created dataset size
	print('dataset size : {}'.format(len(features)))
	return features

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(create database)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--dataset', '-d', default='default_dataset_path',
						help='Directory for creating database')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	args = parser.parse_args()

	data_dir = args.dataset


	# Get a list of pictures
	paths = glob.glob(os.path.join(data_dir, './*/*.png'))
	print(paths)
	assert len(paths) != 0 
	# Create the database
	features = createDatabase(paths, args.gpu, args.model)
	# Save the data of database
	torch.save(features, 'result/feature_mnist_mlp.pt')
	torch.save(paths, 'result/path_mnist_mlp.pt')

if __name__ == '__main__':
	main()
