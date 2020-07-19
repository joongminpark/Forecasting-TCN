import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

# model : wavenet-WN-dropout-residual-sumlayers
class Sumlayers(nn.Module):
    """
    <wavenet_residual_no attention>
    
    n_residue: residue channels
    n_skip: skip channels
    history_day : sequence of day (48 * history_day)
    n_repeat: dilation layer setup (# of dialation repetation)
    """
    def __init__(self, n_residue=32, n_skip=128, history_day=3, n_repeat=1, dropout=0.2, bias=False):
        super(Sumlayers, self).__init__()
        if history_day==3:
            self.dilations = [2**i for i in range(7)] + [2**4]
            self.dilations.sort()
            self.dilations *= n_repeat
        elif history_day==5:
            self.dilations = [2**i for i in range(7)] + [112]
            self.dilations.sort()
            self.dilations *= n_repeat
        else: # history_day==7
            self.dilations = [2**i for i in range(8)] + [80]
            self.dilations.sort()
            self.dilations *= n_repeat
        
        self.from_input = nn.Conv1d(in_channels=18,
                                    out_channels=n_residue,
                                    kernel_size=1,
                                    bias=bias)
        self.conv_sigmoid = nn.ModuleList()
        self.conv_tanh = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.skip_scale = nn.ModuleList()
        self.residue_scale = nn.ModuleList()
        
        for d in self.dilations:
            self.conv_sigmoid.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                out_channels=n_residue,
                                                kernel_size=2,
                                                padding=d,
                                                dilation=d,
                                                bias=bias)))
            self.conv_tanh.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                             out_channels=n_residue,
                                             kernel_size=2,
                                             padding=d,
                                             dilation=d,
                                             bias=bias)))
            self.skip_scale.append(nn.Conv1d(in_channels=n_residue,
                                            out_channels=n_skip,
                                            kernel_size=1,
                                            bias=bias))
            self.dropout.append(nn.Dropout(dropout))
            self.residue_scale.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                 out_channels=n_residue,
                                                 kernel_size=1,
                                                 bias=bias)))
        
        self.decoder_dense = nn.Linear(17, n_skip)
        self.conv_post_1 = nn.Conv1d(in_channels=2*n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=1, kernel_size=1)

    def forward(self, encoder_input, decoder_input):
        output = self.preprocess(encoder_input)
        skip_connections = [] # save for generation purposes
        for s, t, drop, skip_scale, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.dropout, self.skip_scale,
                                                			self.residue_scale, self.dilations):
            output, skip = self.residue_forward(output, s, t, drop, skip_scale, residue_scale, d)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:,:,-1] for s in skip_connections]) 
        output = self.postprocess(output, decoder_input)
        return output

    def preprocess(self, x):
        """
        Description : module for preprocess
        """
        output = x.transpose(1, 2) 
        output = self.from_input(output) 
        return output

    def postprocess(self, x, y):
        """
        Description : module for postprocess
        """
        output = F.relu(x)  
        y = F.relu(self.decoder_dense(y))  
        
        # concat (y + output)
        output = torch.cat((y, output), 1)      
        
        # postprocess
        output = output.unsqueeze(-1) 
        output = F.relu(self.conv_post_1(output)) 
        output = self.conv_post_2(output)  
        output = output.squeeze() 
        return output

    def residue_forward(self, x, conv_sigmoid, conv_tanh, drop, skip_scale, residue_scale, dilation):
        """
        Description : module for residue forward
        """
        output = x   
        output_sigmoid = conv_sigmoid(output)[: ,: , :-dilation] 
        output_tanh = conv_tanh(output)[: ,: , :-dilation] 
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh) 
        output = drop(output)
        skip = skip_scale(output)  
        output = residue_scale(output) 
        output = output + x  
        return output, skip


# model : wavenet-WN-dropout-residual-lastlayer
class Lastlayer(nn.Module):
    """
    <wavenet_residual_no attention>
    
    n_residue: residue channels
    n_skip: skip channels
    history_day : sequence of day (48 * history_day)
    n_repeat: dilation layer setup (# of dialation repetation)
    """
    def __init__(self, n_residue=32, n_skip=128, history_day=3, n_repeat=1, dropout=0.2, bias=False):
        super(Lastlayer, self).__init__()
        if history_day==3:
            self.dilations = [2**i for i in range(7)] + [2**4]
            self.dilations.sort()
            self.dilations *= n_repeat
        elif history_day==5:
            self.dilations = [2**i for i in range(7)] + [112]
            self.dilations.sort()
            self.dilations *= n_repeat
        else: # history_day==7
            self.dilations = [2**i for i in range(8)] + [80]
            self.dilations.sort()
            self.dilations *= n_repeat
        
        self.from_input = nn.Conv1d(in_channels=18,
                                    out_channels=n_residue,
                                    kernel_size=1,
                                    bias=bias)
        self.conv_sigmoid = nn.ModuleList()
        self.conv_tanh = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.residue_scale = nn.ModuleList()
        
        for d in self.dilations:
            self.conv_sigmoid.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                out_channels=n_residue,
                                                kernel_size=2,
                                                padding=d,
                                                dilation=d,
                                                bias=bias)))
            self.conv_tanh.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                             out_channels=n_residue,
                                             kernel_size=2,
                                             padding=d,
                                             dilation=d,
                                             bias=bias)))
            self.dropout.append(nn.Dropout(dropout))
            self.residue_scale.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                 out_channels=n_residue,
                                                 kernel_size=1,
                                                 bias=bias)))
        
        self.encoder_dense = nn.Linear(n_residue, n_skip)
        self.decoder_dense = nn.Linear(17, n_skip)
        self.conv_post_1 = nn.Conv1d(in_channels=2*n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=1, kernel_size=1)

    def forward(self, encoder_input, decoder_input):
        output = self.preprocess(encoder_input)
        for s, t, drop, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.dropout,
                                                self.residue_scale, self.dilations):
            output = self.residue_forward(output, s, t, drop, residue_scale, d)
        
        # sum up skip connections
        output = self.postprocess(output, decoder_input)
        return output

    def preprocess(self, x):
        """
        Description : module for preprocess
        """
        output = x.transpose(1, 2)
        output = self.from_input(output) 
        return output

    def postprocess(self, x, y):
        """
        Description : module for postprocess
        """
        output = x[:,:,-1]  
        output = F.relu(self.encoder_dense(output)) 
        y = F.relu(self.decoder_dense(y)) 
        
        output = torch.cat((y, output), 1)       
        
        # postprocess
        output = output.unsqueeze(-1)  
        output = F.relu(self.conv_post_1(output))
        output = self.conv_post_2(output) 
        output = output.squeeze()
        return output

    def residue_forward(self, x, conv_sigmoid, conv_tanh, drop, residue_scale, dilation):
        """
        Description : module for residue forward
        """
        output = x  
        output_sigmoid = conv_sigmoid(output)[: ,: , :-dilation] 
        output_tanh = conv_tanh(output)[: ,: , :-dilation] 
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh) 
        output = drop(output)
        output = residue_scale(output)  
        output = output + x  
        return output


# model2 : wavenet-WN-dropout-layer attention<tanh + weight> (skip scale)
class Attention(nn.Module):
    """
    <wavenet_residual_no attention>
    
    n_residue: residue channels
    n_skip: skip channels
    history_day : sequence of day (48 * history_day)
    n_repeat: dilation layer setup (# of dialation repetation)
    """
    def __init__(self, n_residue=32, n_skip=128, history_day=3, n_repeat=1, dropout=0.2, bias=False):
        super(Attention, self).__init__()
        if history_day==3:
            self.dilations = [2**i for i in range(7)] + [2**4]
            self.dilations.sort()
            self.dilations *= n_repeat
        elif history_day==5:
            self.dilations = [2**i for i in range(7)] + [112]
            self.dilations.sort()
            self.dilations *= n_repeat
        else: # history_day==7
            self.dilations = [2**i for i in range(8)] + [80]
            self.dilations.sort()
            self.dilations *= n_repeat
        
        self.from_input = nn.Conv1d(in_channels=18,
                                    out_channels=n_residue,
                                    kernel_size=1,
                                    bias=bias)
        self.conv_sigmoid = nn.ModuleList()
        self.conv_tanh = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.skip_scale = nn.ModuleList()
        self.residue_scale = nn.ModuleList()
        
        for d in self.dilations:
            self.conv_sigmoid.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                out_channels=n_residue,
                                                kernel_size=2,
                                                padding=d,
                                                dilation=d,
                                                bias=bias)))
            self.conv_tanh.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                             out_channels=n_residue,
                                             kernel_size=2,
                                             padding=d,
                                             dilation=d,
                                             bias=bias)))
            self.dropout.append(nn.Dropout(dropout))
            self.skip_scale.append(nn.Conv1d(in_channels=n_residue,
                                          out_channels=n_skip,
                                          kernel_size=1,
                                          bias=bias))
            self.residue_scale.append(weight_norm(nn.Conv1d(in_channels=n_residue,
                                                 out_channels=n_residue,
                                                 kernel_size=1,
                                                 bias=bias)))
        
        self.decoder_attn = nn.Linear(17, n_skip)
        self.attn = nn.Linear(n_skip, n_skip)
        self.conv_post_1 = nn.Conv1d(in_channels=2*n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=1, kernel_size=1)

    def forward(self, encoder_input, decoder_input):
        output = self.preprocess(encoder_input)
        skip_connections = [] # save for generation purposes
        for s, t, drop, skip_scale, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.dropout,
                                                          self.skip_scale, self.residue_scale, self.dilations):
            output, skip = self.residue_forward(output, s, t, drop, skip_scale, residue_scale, d)
            skip_connections.append(skip)
        
        # sum up skip connections
        output = [s[:,:,-1] for s in skip_connections]
        output = self.layer_attention(output, decoder_input)  
        output = self.postprocess(output)  
        return output

    def preprocess(self, x):
        """
        Description : module for preprocess
        """
        output = x.transpose(1, 2) 
        output = self.from_input(output) 
        return output

    def postprocess(self, x):
        """
        Description : module for postprocess
        """
        output = F.relu(x.unsqueeze(-1)) 
        output = F.relu(self.conv_post_1(output)) 
        output = self.conv_post_2(output)   
        output = output.squeeze() 
        return output

    def residue_forward(self, x, conv_sigmoid, conv_tanh, drop, skip_scale, residue_scale, dilation):
        """
        Description : module for residue forward
        """
        output = x   
        output_sigmoid = conv_sigmoid(output)[: ,: , :-dilation]
        output_tanh = conv_tanh(output)[: ,: , :-dilation]
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh) 
        output = drop(output)
        skip = skip_scale(output) 
        output = residue_scale(output)  
        output = output + x  
        return output, skip
    
    def layer_attention(self, x, y):
        """
        Description : module for attention (Across layer of end of sequence)
        """
        output = torch.stack(x) 
        output = output.permute(1,0,2)  
        y = F.relu(self.decoder_attn(y)) 
        
        # Attention weights
        attn_y = self.attn(y)   
        attn_y = F.tanh(attn_y.unsqueeze(-1))
        weights = F.softmax(output.bmm(attn_y).squeeze(-1), -1)
        context = weights.unsqueeze(1).bmm(output).squeeze(1) 
        
        output = torch.cat((y, context), 1) 
        
        return output
