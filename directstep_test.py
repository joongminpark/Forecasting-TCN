import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time

from torch.nn.utils import weight_norm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import data_loader
from models import TCN1A, TCN2A, TCN1B, TCN2B


class Sequence_test():
    """
    Description : Sequential data testing for model
    valid_data : type -> numpy [seq, dim]
    """
    def __init__(self, model_type, file_dir, n_residue, n_skip, history_day, n_repeat, dropout, seq_output, dataloaders, scaling_var, device):
        self.model_type, self.file_dir, self.n_residue, self.n_skip, self.history_day, self.n_repeat, self.dropout, self.seq_output = \
        model_type, file_dir, n_residue, n_skip, history_day, n_repeat, dropout, seq_output
        
        # input_data : [B, history_day*48, 18]  /  GT_data : [B, seq_output, 18]
        self.encoder_dataloader = dataloaders['val'][0]
        self.decoder_dataloader = dataloaders['val'][1]
        self.scaling_var = scaling_var
        self.device = device
        
        self.load_model()
        
    
    # file_dir : "./model/Nrepeat1_Wavenet_WN_relu_dropout_residual_layerattention_3days/n_residue32_epoch_80_torch_model"
    def load_model(self):
        self.net = self.model_type(n_residue=self.n_residue, n_skip=self.n_skip, history_day=self.history_day, 
                                   n_repeat=self.n_repeat, dropout=self.dropout, bias=False)
        self.net.load_state_dict(torch.load(self.file_dir))
        self.net.to(self.device)
        self.net.eval()
    
    
    def rmse_mape_test(self):
        """
        Description : RMSE / MAPE (test val_dataset for value of predict day)
        """
        # data : BxSxdim
        GT = []
        y_hat = []
        for en_input, de_input in zip(self.encoder_dataloader, self.decoder_dataloader):
            en_input = en_input.to(self.device)
            de_input = de_input.to(self.device)
            
            tmp_GT = de_input[:,-1,0]
            de_input = de_input[:,-1,1:]
            
            outputs = self.net(en_input, de_input)
            
            # rescaling
            scaling_var = list(self.scaling_var.items())
            
            tmp_GT = tmp_GT*scaling_var[1][1] + scaling_var[0][1]
            tmp_y_hat = outputs*scaling_var[1][1] + scaling_var[0][1]
            
            GT.extend(tmp_GT.tolist())
            y_hat.extend(tmp_y_hat.tolist())
            
        # matrics
        GT = np.array(GT)
        y_hat = np.array(y_hat)
        rmse = np.sqrt(np.mean(np.square((GT - y_hat))))
        mape = np.mean(np.abs((GT - y_hat) / GT)) * 100

        print('RMSE : {}, MAPE : {}'.format(rmse, mape))
        
    

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
	parser.add_argument('--save_path', type=str, default='./model_save/TCN/dropout/dormitory/TCN1A_lastlayer_3days/model_name', help='type : ./model_save/TCN/dormitory/dropout/TCN1A_lastlayer_3days (TCN1A, TCN1B, TCN2A, TCN2B, lastlayer, attention, sumlayers)')

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_loading = data_loader.Data_loader(args.load_path, args.batch_size, args.history_day, args.seq_output)

	test = Sequence_test(eval(args.model_type), args.save_path, args.n_residue, args.n_skip, args.history_day, args.n_repeat, args.dropout, args.seq_output, data_loading.dataloaders, data_loading.scaling_var, device)
	start = time.time()
	test.rmse_mape_test()
	print(time.time() - start)

if __name__ == "__main__":
	main()