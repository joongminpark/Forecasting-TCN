
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader

class Data_loader():
	def __init__(self, file_path, batch_size, history_day, seq_output):
		df = pd.read_csv(file_path)
		df, scaling_var = self.feature_scaling(df)
		df = np.array(df)
		train_data = df[:52608]
		valid_data = df[52608:]
		data = self.devide_train_val(train_data, valid_data, history_day, seq_output=seq_output)
		dataloaders = self.batch_divide(data, batch_size)

		self.dataloaders = dataloaders
		self.scaling_var = scaling_var

	def feature_scaling(self, df, scaling_type='standard', column_idx=[0,9,10,11]):
		"""
		Args:
			scaling_var : [('var_name', mean), ('var_name', std) ...]
			variables(var) : 'Usedpower', 'modiTemp', 'Wind', 'Humi'
		"""    
		scaling_var = []
		
		for idx in column_idx:
			if scaling_type=='standard':
				tmp_mean = np.mean(df.iloc[:52608,idx])
				tmp_std = np.std(df.iloc[:52608,idx])
				
				scaling_var.append((df.columns[idx]+'_mean', tmp_mean))
				scaling_var.append((df.columns[idx]+'_std', tmp_std))
			
				df.iloc[:,idx] = (df.iloc[:,idx] - tmp_mean) / (tmp_std)
				
			elif scaling_type=='minmax':
				tmp_min = np.min(df.iloc[:52608,idx])
				tmp_max = np.max(df.iloc[:52608,idx])

				scaling_var.append((df.columns[idx]+'_min', tmp_min))
				scaling_var.append((df.columns[idx]+'_max', tmp_max))

				df.iloc[:,idx] = (df.iloc[:,idx] - tmp_min) / (tmp_max-tmp_min)
			
		return df, dict(scaling_var)
	
	def create_dataset(self, data, history_day=3, seq_output=48):
		idx = 48*history_day
		num_window = len(data)-seq_output-idx+1
		
		before_data = np.array([data[i:(i+idx)].tolist() for i in range(num_window) 
								if (idx+i+seq_output) <= len(data)-1])
		
		predict_data = np.array([data[(idx+i):(idx+i+seq_output)].tolist() for i in range(num_window) 
								if (idx+i+seq_output) <= len(data)-1])
		
		return torch.from_numpy(before_data).float(), torch.from_numpy(predict_data).float()

	def devide_train_val(self, train_data, valid_data, history_day=3, seq_output=48):
		data = {}

		for x in ['train', 'val']:
			"""
			Args:
				data : {'train' : [encoder_input, decoder_input]
						'val' : [encoder_input, decoder_input]}
				encoder_input, decoder_input : (Num, Seq, variables)
				variables : 'UsedPower','isMon','isTue','isWed','isThu','isFri','isSat','isSun','isHoliday',
						'Wind','Humi','modiTemp', 'Hourmin_X','Hourmin_Y','Month_X','Month_Y','Day_X','Day_Y'
				'Usedpower' : target
			"""    
			data[x] = []
			if x == 'train':
				encoder_input, decoder_input = self.create_dataset(train_data, history_day, seq_output)
				data[x].append(encoder_input)
				data[x].append(decoder_input)

			else:
				encoder_input, decoder_input = self.create_dataset(valid_data, history_day, seq_output)
				data[x].append(encoder_input)
				data[x].append(decoder_input)
							
		return data

	def batch_divide(self, data, batch_size=256):
		dataloaders = {}
		
		for x in ['train', 'val']:
			"""
			Args:
				data : {'train' : [encoder_dataloader, decoder_dataloader]
						'val' : [encoder_dataloader, decoder_dataloader]}
			"""    
			dataloaders[x] = []
			if x == 'train':
				encoder_dataloader = DataLoader(data[x][0], batch_size=batch_size, num_workers=4, drop_last=True)
				dataloaders[x].append(encoder_dataloader)
				decoder_dataloader = DataLoader(data[x][1], batch_size=batch_size, num_workers=4, drop_last=True)
				dataloaders[x].append(decoder_dataloader)
			else:
				encoder_dataloader = DataLoader(data[x][0], batch_size=batch_size, num_workers=4, drop_last=True)
				dataloaders[x].append(encoder_dataloader)
				decoder_dataloader = DataLoader(data[x][1], batch_size=batch_size, num_workers=4, drop_last=True)
				dataloaders[x].append(decoder_dataloader)
				
		return dataloaders
