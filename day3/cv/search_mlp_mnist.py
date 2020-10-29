
import argparse
import torch
from torchvision import transforms
from PIL import Image
from network_db import MLP_DB

def search(src, db_features, db_paths, k, gpu, model_path):
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

	# Get features
	with torch.no_grad():
		src_feature = model(data_preprocess(Image.open(src, 'r')).unsqueeze(0).to(device).view(-1, 28*28)).to('cpu')
		print(src_feature.shape)
	# Load database
	paths = torch.load(db_paths)
	features = torch.load(db_features)
	assert k <= len(paths)
	assert len(features) == len(paths)
	# Calculate distances
	distances = torch.tensor(
		[torch.norm(src_feature - feature)
			for feature in features]
	)
	_, indices = torch.topk(distances, k, largest=False)
	# Show results
	for i in indices:
		print(paths[i])

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(search image)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--input', '-i', default='default_source_image_path',
						help='path to database features')
	parser.add_argument('--features', '-f', default='result/feature_mnist_mlp.pt',
						help='path to database features')
	parser.add_argument('--paths', '-p', default='result/path_mnist_mlp.pt',
						help='path to database paths')
	parser.add_argument('--k', '-k', type=int, default=5,
						help='find num')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	args = parser.parse_args()

	search(args.input, args.features, args.paths, args.k, args.gpu, args.model)

if __name__ == '__main__':
	main()