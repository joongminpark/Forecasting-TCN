import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import weight_norm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src import data_loader, train, TCN1B, TCN2B


def main():
	parser = argparse.ArgumentParser(description='load forecasting')
	parser.add_argument('--batch_size', type=int, default=256, help='batch size in training (default: 256)')
	parser.add_argument('--history_day', type=int, default=3, help='sequence of history day of model (default: 3)')
	parser.add_argument('--seq_output', type=int, default=48, help='sequence of output of model (default: 48)')
	parser.add_argument('--n_residue', type=int, default=32, help='number of TCN residual features of model (default: 32)')
	parser.add_argument('--n_skip', type=int, default=128, help='number of TCN skip features of model (default: 128)')
	parser.add_argument('--n_repeat', type=int, default=1, help='number of total TCN repeats of model (default: 1)')
	parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
	parser.add_argument('--max_epochs', type=int, default=100, help='number of max epochs in training (default: 100)')
	parser.add_argument('--lr', type=float, default=1e-03, help='learning rate (default: 0.001)')
	parser.add_argument('--lr_scheduler', type=int, default=40, help='changing learing rate by step size (default: 40)')
	parser.add_argument('--model_type', type=str, default='TCN1B.Lastlayer', help='which model you choose')
	parser.add_argument('--load_path', type=str, default='./dataset/dormitory_preprocessing.csv', help='file path in loading')
	parser.add_argument('--save_path', type=str, default='./model_save/TCN/dropout/TCN1A_lastlayer_3days', help='type : ./model_save/TCN/dropout/TCN1A_lastlayer_3days (TCN1A, TCN1B, TCN2A, TCN2B, lastlayer, attention, sumlayers)')

	args = parser.parse_args()

	# data loading (batch data : data_loading.dataloaders)
	data_loading = data_loader.Data_loader(args.load_path, args.batch_size, args.history_day, args.seq_output)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = eval(args.model_type)(args.n_residue, args.n_skip, args.history_day, args.n_repeat, args.dropout)
	model.to(device)

	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	
	optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
	criterion = nn.SmoothL1Loss().to(device)
	exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=args.lr_scheduler, gamma=0.1)

	if args.model_type.split('.')[0] in 'TCN1B or TCN2B':
		model = train.trainB(model, data_loading.dataloaders, criterion, optim, exp_lr_scheduler, args.save_path, device, args.max_epochs)
	else:
		model = train.trainA(model, data_loading.dataloaders, criterion, optim, exp_lr_scheduler, args.save_path, device, args.max_epochs)

if __name__ == "__main__":
	main()
