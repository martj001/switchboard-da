import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference: https://arxiv.org/pdf/1807.03748.pdf
class Encoder(nn.Module):
    # x = [batch_size, n_feature, ts_length]
    # y = encoder(x)
    # y = [batch_size, n_features, phi_x_len]
    def __init__(self, n_channel=256, bias=True):
        super().__init__()
        
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, n_channel, kernel_size=10, stride=5, padding=3, bias=bias),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_channel, n_channel, kernel_size=8, stride=4, padding=2, bias=bias),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_channel, n_channel, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_channel, n_channel, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
        )
        self.n_channel = n_channel
    
    def forward(self, x):
        return self.encoder(x)

    
# Reference: https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/5
class MultiHeadDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=512, n_heads=8):
        super().__init__()
        
        self.mlp = nn.Conv1d(in_channels=input_dim*n_heads, out_channels=output_dim*n_heads, kernel_size=1, groups=n_heads)
        self.n_heads = n_heads
      
    def forward(self, x):
        x = x.repeat(1, self.n_heads).unsqueeze(-1)
        flat = self.mlp(x) # flattened out tensor
        batch_size = x.shape[0]
        out = flat.view(batch_size, -1, self.n_heads) # reshape to n_heads outputs
        
        return out
    
    
# Reference: https://arxiv.org/pdf/1807.03748.pdf
class CPC(nn.Module):
    def __init__(
        self,
        phi_dim=256,
        phi_bias=True,
        c_dim=128,
        rnn_num_layers=1,
        decoder_heads=8,
    ):
        super().__init__()
        
        self.phi_dim = phi_dim
        self.phi_bias = phi_bias
        self.encoder = Encoder(n_channel=phi_dim, bias=phi_bias)
        
        self.c_dim = c_dim
        self.rnn_num_layers = rnn_num_layers
        self.auto_regressive = nn.GRU(phi_dim, c_dim, rnn_num_layers)
        
        self.decoder_heads = decoder_heads
        self.decoder = MultiHeadDecoder(input_dim=c_dim, output_dim=phi_dim, n_heads=decoder_heads)
    
    def forward(self, x):
        phi_n = self.encoder(x)
        c_n, h_n = self.auto_regressive(torch.permute(phi_n, (0, 2, 1)))
        
        return phi_n, c_n
    
    
class CLF_Head(nn.Module):
    def __init__(self, input_dim=256, output_dim=10):
        super().__init__()
        
        self.mlp = nn.Sequential( # downsampling factor = 160
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=1),
        )
      
    def forward(self, x):
        out = self.mlp(x)
        
        return out
    
    
class CPC_classifier(nn.Module):
    def __init__(
        self,
        phi_dim=256,
        phi_bias=True,
        c_dim=128,
        rnn_num_layers=1,
        y_dim=10,
    ):
        super().__init__()
        
        self.phi_dim = phi_dim
        self.phi_bias = phi_bias
        self.encoder = Encoder(n_channel=phi_dim, bias=phi_bias)
        
        self.c_dim = c_dim
        self.rnn_num_layers = rnn_num_layers
        self.auto_regressive = nn.GRU(phi_dim, c_dim, rnn_num_layers, batch_first=True)
        
        self.y_dim = y_dim
        self.decoder = CLF_Head(input_dim=c_dim, output_dim=y_dim)
    
    def forward(self, x):
        phi_n = self.encoder(x)
        c_n, h_n = self.auto_regressive(torch.permute(phi_n, (0, 2, 1)))
        y_hat = self.decoder(h_n[0])

        return y_hat
    
    
class CPC_classifier_v3(nn.Module):
    def __init__(
        self,
        phi_dim=512,
        phi_bias=True,
        c_dim=256,
        rnn_num_layers=2,
        y_dim=10,
    ):
        super().__init__()
        
        self.phi_dim = phi_dim
        self.phi_bias = phi_bias
        self.encoder = Encoder(n_channel=phi_dim, bias=phi_bias)
        self.encoder2 = Encoder(n_channel=1, bias=False)
        
        self.c_dim = c_dim
        self.rnn_num_layers = rnn_num_layers
        self.auto_regressive = nn.GRU(phi_dim+1, c_dim, rnn_num_layers, batch_first=True)
        
        self.y_dim = y_dim
        self.decoder = CLF_Head(input_dim=c_dim, output_dim=y_dim)
    
    def forward(self, x):
        phi_n = self.encoder(x[:,0:1,:])
        ind_n = self.encoder2(x[:,1:2,:])
        phi_cat = torch.concat((phi_n,ind_n),1)
        
        c_n, h_n = self.auto_regressive(torch.permute(phi_cat, (0, 2, 1)))
        y_hat = self.decoder(h_n[0])

        return y_hat
    
    
def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


# Reference: https://github.com/RElbers/info-nce-pytorch
def infoNCEloss(query, positive_key, negative_keys, temperature=0.1, reduction='mean'):
    # Normalization
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    # Dot product(query, pos)
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

    # Dot product(query, negative) - pairwise
    negative_logits = query @ transpose(negative_keys)

    # Column 0: p(x_t | c_t), Column 1-n_neg_sample: p(not x_t |  c_t)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device) # first sample is true

    loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
    
    return loss
    
    
# Reference: https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
def cyclical_lr(stepsize, min_lr=0.0001, gap_lr=0.01, exp_decay=1.5e3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + gap_lr * relative(it, stepsize) * np.exp(-it/exp_decay)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda
