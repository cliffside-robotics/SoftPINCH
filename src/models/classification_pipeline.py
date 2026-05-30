# Classification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import sherpa
from sklearn.model_selection import KFold

# Manage datasets
import numpy as np
import pandas as pd

# Preprocessing
from scipy.signal import iirnotch, lfilter, lfilter_zi, butter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d

# Manage utils
import os 
from pathlib import Path
#from datetime import datetime
import logging                  # Avoid loggings from GP
from copy import deepcopy       # Used for copy model_state
import time
import matplotlib.pyplot as plt

from typing import Dict

# Analysis
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
# from statsmodels.stats.contingency_tables import mcnemar

# Own implementations
from src.utilities.preprocessing import EEG_preprocessing, EMG_preprocessing #, RejectBadEpochs, Filtering #E402
from src.utilities.trainer_and_evaluator import FusionNet_train_eval, SingleNet_train_eval
from src.utilities.load_and_visualize_data import load_datasets
from src.utilities.preprocessing import RejectBadEpochs

# Avoid messages for sherpa
logging.getLogger("GP").setLevel(logging.CRITICAL)
logging.getLogger("GPy").setLevel(logging.CRITICAL)

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 8, 9, 10, 11]

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.5
EEG_HIGHCUT = 30

EEG_NUM_CH = len(EEG_USEABLE_CHANNELS)
EMG_NUM_CH = 3

TRIAL_PERIOD = 9
TRIM_PERIOD = 3

RMS_SAMPLING_WINDOW = 500           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 50            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100
HAMPEL_SIGMA = 2

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EMG_CONFIG_DICT = {
    'rms_windowsize' : RMS_SAMPLING_WINDOW,
    'rms_stepsize' : RMS_WINDOW_STEPSIZE,
    'hampel_windowsize' : HAMPEL_WINDOWSIZE,
    'hampel_sigma' : HAMPEL_SIGMA,
    'hampel_plot_option' : [False, None],
    'include_EMG' : False
}

REJECT_CONFIG_DICT = {
    'EEG_epoch_rejection_tolerance' : 6,
    'EMG_epoch_rejection_tolerance' : 6,
    'EEG_ch_acceptance' : 0,
    'EMG_ch_acceptance' : 0
}
    # Tolerance -> RANGE given [6 : 8]
    # EMG -> RANGE given by [0, 1]
    # EEG all CH -> RANGE given [0 : 3]
    # EEG 6 CH -> RANGE given [0 : 2]

#========#
# Models #
#========#
class LSTM(nn.Module):
    '''
    Perform Long-short-term-memory on data

    Parameters
    ----------
    input_dim : int
        Begin dimension of data channels if performed alone.
        With CNN the input is set to CNN_filters
    include_attn : bool
        If True, the return is given by the whole LSTM sequence
        If False, the return is given by the last hidden state units 
    '''
    def __init__(self,
                 input_dim : int,
                 hidden_dim : int,
                 layers : int,
                 dropout : float,
                 bidirectional : bool):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim,            # Don't consider bidirectional for hidden units. 
                            num_layers = layers, 
                            batch_first = True,                  # Data input will be (batch, seq_len, channels)
                            dropout = dropout if layers > 1 else 0,
                            bidirectional = bidirectional)       

    def forward(self, data : torch.Tensor):
        '''
        Perform LSTM

        Parameters
        ----------
        data : torch.Tensor
            Data with shape: (B, S, C) or if CNN (B, S/4, C_filters)
        
        Returns
        ----------
        lstm_units : torch.Tensor
            If include_attn True : units with shape: (B, S, D * hidden_dim), D = 2 if bidirectional
            If include_attn False : units with shape: (B, D * hidden_dim), D = 2 if bidirectional
        '''
        lstm_out, (hn, _) = self.lstm(data)
            # lstm_out : (B, S, D*H)    -> Each time step t, give hidden state of the final layer
            # hn : (D*layers, B, H)     -> For each layer, give last hidden state

        #lstm_units = lstm_out if self.include_attn else self._extract_hidden(units = hn)
        hn = self._extract_hidden(units = hn)

        return lstm_out, hn
    
    def _extract_hidden(self, units : torch.Tensor):
        """
        Extract final hidden state from LSTM
        
        Parameters
        ----------
        units : torch.Tensor
            Final hidden state units shape (D*layers, B, H)
        """
        if self.bidirectional:
            h_forward = units[-2]                                          # (batch, hidden)
            h_backward = units[-1]                                         # (batch, hidden)
            h_final = torch.cat((h_forward, h_backward), dim=1)            # Concat -> (batch, hidden * 2)
        else:
            h_final = units[-1]                                            # (batch, hidden)

        return h_final

class CNN(nn.Module):
    def __init__(self,
                 input_dim : int,
                 cnn_filters : int,
                 kernel_size : int,
                 activation : nn.Module,
                 dropout : float):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = cnn_filters, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters),
            activation(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = cnn_filters, out_channels = cnn_filters*2, kernel_size = kernel_size, padding = kernel_size // 2),        
            nn.BatchNorm1d(num_features = cnn_filters*2),
            activation(),           
            nn.MaxPool1d(kernel_size = 2),
            
            nn.Dropout(dropout)
            
        )
    
    def forward(self, data : torch.Tensor):
        '''
        Perform Convolutional neural network

        Parameters
        ----------
        data : torch.Tensor
            Data to be forwarded into the neural network
        
        Returns
        ----------
        x : torch.Tensor
            Spatial features with shape : (B, S/4, C_filt)
        '''
        x = data.permute(0, 2, 1)      # (B, S, CH) -> (B, CH, S)
        x = self.cnn(x)                # (B, CH, S) -> (B, C_filter, S/4)
        x = x.permute(0, 2, 1)         # (B, C_filter, S/4) -> (B, S/4, C_filter)

        return x

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))           # Learnable vector to scalar score for each time step. Weights are updated during backpropagation

        # 🔹 Learnable query vector
        # self.query = nn.Parameter(torch.randn(hidden_dim))
        # 🔹 Expand query to match sequence
        # q = self.query.unsqueeze(0).unsqueeze(0)   # (1, 1, H)
        # Q = q.expand(B, S, H)                      # (B, S, H)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        Q = hidden.unsqueeze(1)                                      # (B, H) -> (B, 1, H) - Prepare the decoder hidden state as the query for attention
        K = encoder_outputs                                          # (B, S, H) - Encoder hidden states serve as keys for attention

        Q = Q.expand(-1, seq_len, -1)                                # (B, 1, H) -> (B, seq_len, H)

        #=======================================================================================#
        # Step 1)  Feed-Forward Alignment Function: The decoder’s current hidden state 'S_t'    #
        # and each encoder hidden state 'h_i' are combined to compute alignment scores 'e_t,i'. #
        # energy (e_t,i) = v^T * tanh(W_a * [S_t; h_i]) where S_t is query and h_i is keys      #
        #=======================================================================================#
        QK = torch.cat((Q, K), dim=2)         # (B, S, 2H)
        energy = torch.tanh(self.attn(QK))                       # (B, S, H) - Learns a transformation from the concatenated decoder-hidden + encoder-output to an intermediate "energy" vector
        energy = energy.permute(0, 2, 1)                            # (B, H, S) - Permute for batch matrix multiplication

        #==================================#
        # Step 2) Compute attention scores #
        #==================================#
        v = self.v.repeat(batch_size, 1).unsqueeze(1)               # (H) -> (B, H) -> (B, 1, H) - Expand v to match batch size and prepare for batch matrix multiplication
        scores = torch.bmm(v, energy)                               # (B, 1, H) x (B, H, S) -> (B, 1, S) - Compute attention scores for each encoder hidden state
        scores = scores.squeeze(1)                                  # (B, S) - Remove the extra dimension
    
        #===================================================#
        # 3) Convert scores to probabilities using softmax, #
        # yielding attention weights 'a_t,i'                #
        # that sum to 1 across all encoder time steps.      #
        #===================================================#
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), K)  # (B, 1, S) × (B, S, H) -> (B, 1, H) - Compute the context vector as a weighted sum of encoder hidden states

        return context, weights
    
'''Old Attention implementation
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lstm_out):
        """
        Parameter
        ---------
        lstm_out : Tensor
            Output from LSTM with shape (batch, seq_len, hidden)
        
        Return
        --------
        context : Tensor
            Weighted sum over time (batch, hidden)
        
        weights : Tensor
            attension weights (batch, seq_len, 1)
        """
        # Compute attension scores
        scores = self.attn(lstm_out)                 # (B, S, H) -> (B, S, 1)
        
        weights = torch.softmax(scores, dim = 1)     # Apply softmax over 'seq_len' - (B, S, 1)

        # Weigted sum over time - NOTE: The context vector summarizes the most relevant information from the input sequence and is fed to the decoder.
        context = torch.sum(input = weights * lstm_out, dim = 1)    # (B, H)

        return context, weights'''

class DenseLayer(nn.Module):
    '''
    Perform a fully connected layer. Going from LSTM hidden units to number of classes to predict

    Parameters
    ----------
    lstm_hidden_dim : int
        Number of hidden units from lstm
    dense_layers : int
        Linear transform from hiden units to dense_layers
    activation : nn.Module
        Activation function introducing non-linearity to learn complex non-linear patterns in data
    dropout : float
        Randomly set neurons to zero during training
    
    Returns
    ---------
    logits : torch.Tensor
        Logits are raw, non-normalized, unbounded scores ('guesses')
        how confidence the signal belong to seach possible output classes
    '''
    def __init__(self,
                 lstm_hidden_dim : int,
                 output_dim : int,
                 dense_ratio : float,
                 activation : nn.Module,
                 dropout : float):
        super().__init__()

        dense_layers = max(8, int(lstm_hidden_dim * dense_ratio))

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, dense_layers),            
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dense_layers, output_dim)
        )
    
    def forward(self, lstm_units : torch.Tensor):
        return self.classifier(lstm_units)

#==========================#
# Single modality networks #
#==========================#
class SingleNet_LSTM(nn.Module):

    '''
    Single network to perform LSTM on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    '''
    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float):
        super().__init__()
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.lstm = LSTM(input_dim = input_dim,
                         hidden_dim = hidden_dim,
                         layers = lstm_layers,
                         dropout = dropout,
                         bidirectional = bidirectional)
        
        self.classifier = DenseLayer(lstm_hidden_dim = lstm_hidden_dim,
                                     output_dim = output_dim,
                                     dense_ratio = dense_ratio,
                                     activation = act,
                                     dropout = dropout)

    def forward(self, data : torch.Tensor) -> tuple[torch.Tensor, None, None]:
        '''
        Runs the data through LSTM + Fully Connected layer

        Parameters
        ----------
        data : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels) for either EEG or EMG
        
        Returns
        ----------
        logits : torch.Tensor
            Linear transform of LSTM features -> logits
        _ : None
            Placeholder
        _ : None
            Placeholder
        '''
        _, hn = self.lstm(data)

        logits = self.classifier(hn)

        return logits, None, None

class SingleNet_CNN_LSTM(nn.Module):
    '''
    Single network to perform CNN + LSTM on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    '''
    def __init__(self, 
                 input_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float,
                 cnn_filters : int,
                 kernel_size : int):
        super().__init__()
        '''
        Args:
            input_dim - int
                The number of expected features in the input sequence at each time step (n_channels)
            hidden_dim - int
                The number of features in the hidden state. How much memory should present the hidden state at one time stamp
            layer_dim - int
                Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
            output_dim - int
                Maps the hidden state in nn.Linear outputs to predictions (n_classes)
        '''
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.cnn = CNN(input_dim = input_dim,
                       cnn_filters = cnn_filters,
                       kernel_size = kernel_size,
                       activation = act,
                       dropout = dropout)

        self.lstm = LSTM(input_dim = cnn_filters*2,
                         hidden_dim = hidden_dim,
                         layers = lstm_layers,
                         dropout = dropout,
                         bidirectional = bidirectional)
        
        self.classifier = DenseLayer(lstm_hidden_dim = lstm_hidden_dim,
                                     output_dim = output_dim,
                                     dense_ratio = dense_ratio,
                                     activation = act,
                                     dropout = dropout)

    def forward(self, data : torch.Tensor):
        '''
        Runs the data through CNN + LSTM + Fully Connected layer

        Parameters
        ----------
        data : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels) for either EEG or EMG
        
        Returns
        ----------
        logits : torch.Tensor
            Linear transform of LSTM features -> logits
        _ : None
            Placeholder
        _ : None
            Placeholder
        '''
        cnn_out = self.cnn(data)               

        _, hn = self.lstm(cnn_out)        

        logits = self.classifier(hn)

        return logits, None, None        # None is placeholders

class SingleNet_CNN_LSTM_ATTENTION(nn.Module):
    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float,
                 cnn_filters : int,
                 kernel_size : int):
        super().__init__()
        '''
        Args:
            input_dim - int
                The number of expected features in the input sequence at each time step (n_channels)
            hidden_dim - int
                The number of features in the hidden state. How much memory should present the hidden state at one time stamp
            layer_dim - int
                Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
            output_dim - int
                Maps the hidden state in nn.Linear outputs to predictions (n_classes)
        '''
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim  

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.cnn = CNN(input_dim = input_dim,
                       cnn_filters = cnn_filters,
                       kernel_size = kernel_size,
                       activation = act,
                       dropout = dropout)

        self.lstm = LSTM(input_dim = cnn_filters*2,
                         hidden_dim = hidden_dim,
                         layers = lstm_layers,
                         dropout = dropout,
                         bidirectional = bidirectional)
        
        self.attn = Attention(hidden_dim = lstm_hidden_dim)
        
        self.classifier = DenseLayer(lstm_hidden_dim = lstm_hidden_dim,
                                     output_dim = output_dim,
                                     dense_ratio = dense_ratio,
                                     activation = act,
                                     dropout = dropout)


    def forward(self, data : torch.Tensor):
        cnn_out = self.cnn(data)               

        lstm_units, hn = self.lstm(cnn_out)        

        context, attn_weights = self.attn(hn, lstm_units)

        context = context.squeeze(1)

        logits = self.classifier(context)           

        return logits, context, attn_weights

#================#
# Fusion network #
#================#
class FusionNet_LSTM(nn.Module):
    def __init__(self, 
                 eeg_dim : int,
                 emg_dim : int,
                 eeg_output_dim : int, 
                 emg_output_dim : int, 
                 output_dim : int, 
                 hidden_dim : int, 
                 lstm_layers : int, 
                 bidirectional : bool, 
                 dropout : float, 
                 activation : str, 
                 dense_ratio : float):
        super().__init__()

        #=========================================================================#
        # NOTE: Only used when EMG has no bidirectional and 1 lstm layer          #
        # Else:                                                                   #
        #   dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim #
        #   dense_layers = max(8, int(dense_hidden_layers * dense_ratio))         #
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        #   nn.Linear(dense_hidden_layers, dense_layers)                          #            
        #   nn.Linear(dense_layers, X_output_dim)                                 #
        #=========================================================================#
        eeg_lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim  

        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]


        self.eeg_lstm = LSTM(input_dim = eeg_dim,
                             hidden_dim = hidden_dim,
                             layers = lstm_layers,
                             dropout = dropout,
                             bidirectional = bidirectional)

        self.emg_lstm = LSTM(input_dim = emg_dim,
                             hidden_dim = hidden_dim,
                             layers = 1,
                             dropout = dropout,
                             bidirectional = False)

        self.eeg_dense = DenseLayer(lstm_hidden_dim = eeg_lstm_hidden_dim,
                                     output_dim = eeg_output_dim,
                                     dense_ratio = dense_ratio,
                                     activation = act,
                                     dropout = dropout)
        
        self.emg_dense = DenseLayer(lstm_hidden_dim = hidden_dim,
                                    output_dim = emg_output_dim,
                                    dense_ratio = dense_ratio,
                                    activation = act,
                                    dropout = dropout)
        
        fusion_input_dim = eeg_output_dim + emg_output_dim

        self.fusion = DenseLayer(lstm_hidden_dim = fusion_input_dim,
                                 output_dim = output_dim,
                                 dense_ratio = 1.0,
                                 activation = act,
                                 dropout = dropout)

    def forward(self, eeg : torch.Tensor, emg : torch.Tensor):

        # EEG branch
        _, eeg_lstm_units = self.eeg_lstm(eeg)
        eeg_logits = self.eeg_dense(eeg_lstm_units)

        # EMG branch
        _, emg_lstm_units = self.emg_lstm(emg)
        emg_logits = self.emg_dense(emg_lstm_units)

        # Late fusion
        fusion_input = torch.cat([eeg_logits, emg_logits], dim=1)

        # Normalize logits
        f_mu, f_std = torch.mean(fusion_input), torch.std(fusion_input)
        fusion_input = (fusion_input - f_mu) / (f_std + 1e-8)

        fusion_logits = self.fusion(fusion_input)

        return fusion_logits, eeg_logits, emg_logits

class FusionNet_CNN_LSTM(nn.Module):
    '''
    Fusion network to perform CNN + LSTM on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    eeg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    emg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    eeg_output_dim : int
        The expected amount of classes to be predicted
    emg_output_dim : int
        The expected amount of classes to be predicted
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    activation : str (Optional hyperparameter)
        Activation function used in the model
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    dense_fusion_layer : int
        Final dense layer given by eeg_output_dim + emg_output_dim -> dense_fusion_layer
    cnn_filters : int (Optional hyperparameter)
        Filters in the CNN 
    kernel_size : int
        Kernel size in the CNN
    '''
    def __init__(self,
                 eeg_dim : int,
                 emg_dim : int,
                 eeg_output_dim : int,
                 emg_output_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float,
                 cnn_filters : int,
                 kernel_size : int):
        super().__init__()
        #=========================================================================#
        # NOTE: Only used when EMG has no bidirectional and 1 lstm layer          #
        # Else:                                                                   #
        #   dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim #
        #   dense_layers = max(8, int(dense_hidden_layers * dense_ratio))         #
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        #   nn.Linear(dense_hidden_layers, dense_layers)                          #            
        #   nn.Linear(dense_layers, X_output_dim)                                 #
        #=========================================================================#
        eeg_lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.eeg_cnn = CNN(input_dim = eeg_dim,
                           cnn_filters = cnn_filters,
                           kernel_size = kernel_size, 
                           activation = act,
                           dropout = dropout)

        self.emg_cnn = CNN(input_dim = emg_dim,
                           cnn_filters = cnn_filters,
                           kernel_size = kernel_size, 
                           activation = act,
                           dropout = dropout)
        
        self.eeg_lstm = LSTM(input_dim = cnn_filters*2,
                             hidden_dim = hidden_dim,
                             layers = lstm_layers,
                             dropout = dropout,
                             bidirectional = bidirectional)

        self.emg_lstm = LSTM(input_dim = cnn_filters*2,
                             hidden_dim = hidden_dim,
                             layers = 1,
                             dropout = dropout,
                             bidirectional = False)

        self.eeg_dense = DenseLayer(lstm_hidden_dim = eeg_lstm_hidden_dim,
                                    output_dim = eeg_output_dim,
                                    dense_ratio = dense_ratio,
                                    activation = act,
                                    dropout = dropout)

        self.emg_dense = DenseLayer(lstm_hidden_dim = hidden_dim,
                                    output_dim = emg_output_dim,
                                    dense_ratio = dense_ratio,
                                    activation = act,
                                    dropout = dropout)
        
        fusion_input_dim = eeg_output_dim + emg_output_dim

        self.fusion = DenseLayer(lstm_hidden_dim = fusion_input_dim,
                                 output_dim = output_dim,
                                 dense_ratio = 1.0,
                                 activation = act,
                                 dropout = dropout)
    
    def forward(self, eeg : torch.Tensor, emg : torch.Tensor):
        '''
        Runs the data through CNN + LSTM + Fully Connected layer

        Parameters
        ----------
        eeg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)
        emg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)

        Returns
        ----------
        fusion_logits : torch.Tensor
            Linear transform of LSTM features for eeg and emg -> logits
        eeg_logits : torch.Tensor
            Linear transform of LSTM features for eeg -> logits
        emg_logits : torch.Tensor
            Linear transform of LSTM features for emg -> logits
        '''
        # EEG branch
        eeg_cnn = self.eeg_cnn(eeg)
        _, eeg_lstm_units = self.eeg_lstm(eeg_cnn)
        eeg_logits = self.eeg_dense(eeg_lstm_units)

        # EMG branch
        emg_cnn = self.emg_cnn(emg)
        _, emg_lstm_units = self.emg_lstm(emg_cnn)
        emg_logits = self.emg_dense(emg_lstm_units)

        # Late fusion
        fusion_input = torch.cat([eeg_logits, emg_logits], dim=1)

        # Normalize logits
        f_mu, f_std = torch.mean(fusion_input), torch.std(fusion_input)
        fusion_input = (fusion_input - f_mu) / (f_std + 1e-8)

        fusion_logits = self.fusion(fusion_input)

        return fusion_logits, eeg_logits, emg_logits

class FusionNet_CNN_LSTM_ATTENTION(nn.Module):
    '''
    Fusion network to perform CNN + LSTM + attention on EEG or EMG datasets.\n
    Returns logits from LSTM features to number of classes to classify

    Parameters
    ----------
    eeg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    emg_dim : int
        The number of expected features in the input sequence at each time step (n_channels)
    eeg_output_dim : int
        The expected amount of classes to be predicted
    emg_output_dim : int
        The expected amount of classes to be predicted
    output_dim : int
        Maps the hidden state in nn.Linear outputs to predictions (n_classes)
    hidden_dim : int (Optional hyperparameter)
        The number of features in the hidden state. How much memory should present the hidden state at one time stamp
    lstm_layers : int (Optional hyperparameter)
        Stacking multiple LSTM layers deepens the model. If is not in the timestamp direction. But the hiddenstate goes into the input of another LSTM.
    bidirectional : bool (Optional hyperparameter)
        Enable bi-directional LSTM. Doubles the hidden_dim dimensionality.
    dropout : float (Optional hyperparameter)
        Introduce a dropout probability on the outputs of each LSTM layers. Except for the final layer.
    activation : str (Optional hyperparameter)
        Activation function used in the model
    dense_ratio : float (Optional hyperparameter)
        Embed higher-dimensional space (dense_hidden_layers < dense_layers) -> learns more complex nonlienar combinations -> risk of overfitting.
        Distill LSTM features into a compact representation (dense_hidden_layers > dense_layers) -> strong regularization -> risk of underfit and removal of important patterns
    dense_fusion_layer : int
        Final dense layer given by eeg_output_dim + emg_output_dim -> dense_fusion_layer
    cnn_filters : int (Optional hyperparameter)
        Filters in the CNN 
    kernel_size : int
        Kernel size in the CNN
    '''
    def __init__(self,
                 eeg_dim : int,
                 emg_dim : int,
                 eeg_output_dim : int,
                 emg_output_dim : int,
                 output_dim : int,
                 hidden_dim : int,
                 lstm_layers : int,
                 bidirectional : bool,
                 dropout : float,
                 activation : str,
                 dense_ratio : float,
                 cnn_filters : int,
                 kernel_size : int):
        super().__init__()
        #=========================================================================#
        # NOTE: Only used when EMG has no bidirectional and 1 lstm layer          #
        # Else:                                                                   #
        #   dense_hidden_layers = hidden_dim * 2 if bidirectional else hidden_dim #
        #   dense_layers = max(8, int(dense_hidden_layers * dense_ratio))         #
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        #   nn.Linear(dense_hidden_layers, dense_layers)                          #            
        #   nn.Linear(dense_layers, X_output_dim)                                 #
        #=========================================================================#
        eeg_lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        activations = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU
        }
        act = activations[activation]

        self.eeg_cnn = CNN(input_dim = eeg_dim,
                           cnn_filters = cnn_filters,
                           kernel_size = kernel_size, 
                           activation = act,
                           dropout = dropout)

        self.emg_cnn = CNN(input_dim = emg_dim,
                           cnn_filters = cnn_filters,
                           kernel_size = kernel_size, 
                           activation = act,
                           dropout = dropout)
        
        self.eeg_lstm = LSTM(input_dim = cnn_filters*2,
                             hidden_dim = hidden_dim,
                             layers = lstm_layers,
                             dropout = dropout,
                             bidirectional = bidirectional)

        self.emg_lstm = LSTM(input_dim = cnn_filters*2,
                             hidden_dim = hidden_dim,
                             layers = 1,
                             dropout = dropout,
                             bidirectional = False)
        
        self.eeg_attn = Attention(hidden_dim = eeg_lstm_hidden_dim)

        self.emg_attn = Attention(hidden_dim = hidden_dim)

        self.eeg_dense = DenseLayer(lstm_hidden_dim = eeg_lstm_hidden_dim,
                                    output_dim = eeg_output_dim,
                                    dense_ratio = dense_ratio,
                                    activation = act,
                                    dropout = dropout)

        self.emg_dense = DenseLayer(lstm_hidden_dim = hidden_dim,
                                    output_dim = emg_output_dim,
                                    dense_ratio = dense_ratio,
                                    activation = act,
                                    dropout = dropout)
        
        fusion_input_dim = eeg_output_dim + emg_output_dim

        self.fusion = DenseLayer(lstm_hidden_dim = fusion_input_dim,
                                 output_dim = output_dim,
                                 dense_ratio = 1.0,
                                 activation = act,
                                 dropout = dropout)
    
    def forward(self, eeg : torch.Tensor, emg : torch.Tensor):
        '''
        Runs the data through CNN + LSTM + Attention + Fully Connected layer

        Parameters
        ----------
        eeg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)
        emg : torch.Tensor
            DataLoader tensor with dimension: (batch, seq_len, channels)

        Returns
        ----------
        fusion_logits : torch.Tensor
            Linear transform of LSTM features for eeg and emg -> logits
        eeg_logits : torch.Tensor
            Linear transform of LSTM features for eeg -> logits
        emg_logits : torch.Tensor
            Linear transform of LSTM features for emg -> logits
        '''
        # EEG branch
        eeg_cnn = self.eeg_cnn(eeg)
        eeg_lstm_units, eeg_hn = self.eeg_lstm(eeg_cnn)
        eeg_context, _ = self.eeg_attn(eeg_hn, eeg_lstm_units)
        eeg_context = eeg_context.squeeze(1)
        eeg_logits = self.eeg_dense(eeg_context)

        # EMG branch
        emg_cnn = self.emg_cnn(emg)
        emg_lstm_units, emg_hn = self.emg_lstm(emg_cnn)
        emg_context, _ = self.emg_attn(emg_hn, emg_lstm_units)
        emg_context = emg_context.squeeze(1)
        emg_logits = self.emg_dense(emg_context)

        # Late fusion
        fusion_input = torch.cat([eeg_logits, emg_logits], dim=1)

        # Normalize logits
        f_mu, f_std = torch.mean(fusion_input), torch.std(fusion_input)
        fusion_input = (fusion_input - f_mu) / (f_std + 1e-8)

        fusion_logits = self.fusion(fusion_input)

        return fusion_logits, eeg_logits, emg_logits

#=================#
# Handles dataset #
#=================#
class SingleManageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, data_type):
        '''
        Takes in the concatinated dataset of all trials, samples and channels.
        Args:
            X [ndArray] - with the dimension of (trials, samples, channels)
            y [int] - Indicate the number of trials 
        '''
        if data_type == 'EEG':
            labels = self._map_to_emg_labels(labels = labels)
        elif data_type == 'EMG':
            labels = labels
        elif data_type == 'BCI_IV_2a':
            labels = labels
        else:
            raise ValueError('data_type must be either EEG or EMG')
        
        # Convert to tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # print('data shape:', self.data.shape)
        # print('labels shape:', self.labels.shape)
        # print()
    
    def _map_to_emg_labels(self, labels):
        '''
        Only applied to EEG dataset
        '''
        map_labels = labels.copy()
        rest_label = labels.max()

        # Rest
        map_labels[labels == rest_label] = 2

        # Contract (even indices)
        map_labels[(labels % 2 == 0) & (labels != rest_label)] = 0

        # Release (odd indices)
        map_labels[(labels % 2 == 1)] = 1

        return map_labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MultiManageDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, emg, eeg_labels, emg_labels):
        map_labels = self._map_to_emg_labels(eeg_labels)

        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.emg = torch.tensor(emg, dtype=torch.float32)
        self.eeg_labels = torch.tensor(map_labels, dtype=torch.long)
        self.emg_labels = torch.tensor(emg_labels, dtype=torch.long)

        # print('eeg shape:', self.eeg.shape)
        # print('emg shape:', self.emg.shape)
        # print('eeg labels shape:', self.eeg_labels.shape)
        # print('emg labels shape:', self.emg_labels.shape)
    
    def _map_to_emg_labels(self, labels):
    
        map_labels = labels.copy()

        map_labels[(labels == 0) | (labels == 2)] = 0
        map_labels[(labels == 1) | (labels == 3)] = 1
        map_labels[labels == 4] = 2

        return map_labels
    
    def __len__(self):
        return len(self.eeg_labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.emg[idx], self.eeg_labels[idx], self.emg_labels[idx]

class Manage3Split:
    '''
    Functionality for splitting continous data into train-validation-test split and\n
    segment trial into rest, contract and release data
    '''
    def __init__(self, seed : int):
        '''
        Parameter
        ---------
        seed : int
            For np.random generator
        '''
        self.rng = np.random.default_rng(seed)
    
    def build_modality_split(self, epoch_dict: Dict, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Random shuffle indicies for trials and divide into train, validation and test dataset.

        Parameters
        ----------
        num_index_trials : int
            Number of trials from X_epoch_index.shape[0]
        num_thumb_trials : int
            Number of trials from X_epoch_thumb.shape[0]
        epoch_index : np.ndarray
            Array of data given in epochs. Shape (Epochs, sequence, channels)
        epoch_thumb : np.ndarray
            Array of data given in epochs. Shape (Epochs, sequence, channels)
        fs : int
            Sampling frequency for either EEG or EMG

        Returns
        ----------
        X_train : np.ndarray
        X_val : np.ndarray 
        X_test : np.ndarray
        y_train : np.ndarray
        y_val : np.ndarray
        y_test : np.ndarray 
        '''
        
        split_indices = {}

        for motion, data in epoch_dict.items():
            num_trials = data.shape[0]
            split_indices[motion] = self._split_trials(num_trials, train_ratio = 0.7)

        # Build datasets
        X_train, y_train = self._build_split(epoch_dict, split_indices, split_type='train', fs=fs)
        X_val, y_val     = self._build_split(epoch_dict, split_indices, split_type='val', fs=fs)
        X_test, y_test   = self._build_split(epoch_dict, split_indices, split_type='test', fs=fs)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_split(self, epoch_dict: dict, split_indices: dict, split_type: str, fs: int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Provides a dataset and labels with 5 classes for the train-validation-test split

        Parameters
        ----------
        epoch_index : np.ndarray
            Epoch data for either EEG or EMG for index motion
        
        epoch_thumb : np.ndarray
            Epoch data for either EEG or EMG for thumb motion

        index_trials_indices : list
            Indices for a specfic split (index). Provided by split_trials function

        thumb_trials_indices : list
            Indices for a specfic split (thumb). Provided by split_trials function
        
        fs : int
            Sampling frequency

        data_type : str
            Either EEG or EMG. Purpose -> To define different amount of classes

        Returns
        ----------
        X : np.ndarray
            Dataset split for either train, validation or test with all 5 classes
        
        y : np.ndarray
            Corresponding labels
        '''

        X_list = []
        y_list = []

        label_counter = 0
        rest_list = []

        for motion, data in epoch_dict.items():
            train_idx, val_idx, test_idx = split_indices[motion]

            if split_type == 'train':
                indices = train_idx
            elif split_type == 'val':
                indices = val_idx
            else:
                indices = test_idx

            selected_trials = data[indices]

            rest, contract, release = self._segment_trials(selected_trials, fs)

            # Contract
            X_list.append(contract)
            y_list.append(np.full(len(contract), label_counter))
            label_counter += 1

            # Release
            X_list.append(release)
            y_list.append(np.full(len(release), label_counter))
            label_counter += 1

            # Collect rest (shared class later)
            rest_list.append(rest)

        # Combine rest from all motions into ONE class
        rest_all = np.concatenate(rest_list)
        X_list.append(rest_all)
        y_list.append(np.full(len(rest_all), label_counter))  # last class = rest

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        return X, y

    def _split_trials(self, num_trials : int, train_ratio : int = 0.7) -> tuple[list, list, list]:
        '''
        Provide indices for a train-validation-test split. Used per finger motion

        Parameters
        ----------
        num_trials : int
            Number of trials per finger motion

        train_ratio : int = 0.7
            Percent ratio for train split
        
        val_ratio : int = 0.15
            Percent ratio for validation and test split
        
        Returns
        ----------
        train_idx : list
            Indices for train split 
        
        val_idx : list
            Indices for validation split
        
        test_idx : list
            Indices for test split
        '''
        indices = self.rng.permutation(num_trials)

        n_train = int(train_ratio * num_trials)
        n_remain = (num_trials - n_train) // 2          # Split test and val equally 

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_remain]
        test_idx  = indices[n_train + n_remain:]

        print(f'Train split indicies {train_idx.shape}\n',
              f'Valdiation split indicies {val_idx.shape}\n',
              f'Test split indicies {test_idx.shape}\n')

        return train_idx, val_idx, test_idx

    def _segment_trials(self, trials : np.ndarray, fs : int) -> tuple[list, list, list]:
        """
        Convert the split data into 3 classes

        Parameters
        ----------

        trials : np.ndarray
            Splited data of shape (num_trials, total_samples, channels)
        
        Returns
        ---------
        rest, contract, release : list
            Segment trial into the 3 classes
        """

        rest     = trials[:, :3*fs, :]
        contract = trials[:, 3*fs:6*fs, :]
        release  = trials[:, 6*fs:, :]

        return rest, contract, release
    
    def build_dataset_from_subjects(self, X_epoch, subjects, fs):
        '''
        Only used for subject-independent classificaiton
        '''
        X_list = []
        y_list = []

        label_counter = 0
        rest_list = []

        for motion in X_epoch[subjects[0]].keys():

            all_trials = []

            # Collect trials across subjects
            for subj in subjects:
                all_trials.append(X_epoch[subj][motion])

            all_trials = np.concatenate(all_trials, axis=0)

            # Use YOUR segmentation
            rest, contract, release = self._segment_trials(all_trials, fs)

            # Contract
            X_list.append(contract)
            y_list.append(np.full(len(contract), label_counter))
            label_counter += 1

            # Release
            X_list.append(release)
            y_list.append(np.full(len(release), label_counter))
            label_counter += 1

            rest_list.append(rest)

        # Shared rest
        rest_all = np.concatenate(rest_list)
        X_list.append(rest_all)
        y_list.append(np.full(len(rest_all), label_counter))

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        return X, y

    def build_dataset_window_relabel(self, X_epoch, X_labels, subjects, split):
        '''
        Only used for subject-dependent classification for real-time applicaiton
        Uses window segments with corresponding labels
        '''
        X_list = []
        y_list = []
        rest_list = []

        label_counter = 0
        classes_increment = 2           # Used to increase labels for motions (contract and release). 

        for motion in X_epoch[subjects[0]].keys():
            for subj in subjects:

                data = X_epoch[subj][motion][split]
                labels = X_labels[subj][motion][split]
                
                # Indicies for classes
                rest_mask = np.where(labels == 'rest')[0]
                contract_mask = np.where(labels == 'contract')[0]
                release_mask = np.where(labels == 'release')[0]

                # Contract
                contract = data[contract_mask]
                X_list.append(contract)
                y_list.append(np.full(len(contract), 0 + label_counter))

                # Release
                release = data[release_mask]
                X_list.append(release)
                y_list.append(np.full(len(release), 1 + label_counter))

                # Rest
                rest = data[rest_mask]
                rest_list.append(rest)

            label_counter += classes_increment

        last_label = max(y_list[-1]) + 1            # Extract last label, add 1 to include rest label

        rest_all = np.concatenate(rest_list)
        X_list.append(rest_all)
        y_list.append(np.full(len(rest_all), last_label))

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        return X, y

class KFoldManageDataset(torch.utils.data.Dataset):
    def __init__(self):
        '''
        Takes in the concatinated dataset of all trials, samples and channels.
        Args:
            X [ndArray] - with the dimension of (trials, samples, channels)
            y [int] - Indicate the number of trials 
        '''
        # Convert to tensors
        # self.data = torch.tensor(data, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.long)

        # print('data shape:', self.data.shape)
        # print('labels shape:', self.labels.shape)
        # print()
    
    def create_kfold_splits_subject_independent(self, subject_ids, k=5):
        '''
        Extract indicies for train and validation sets based on subject IDs.
        Provide K-folds of splits for subject-independent validation. Each fold contains unique subject IDs in the train and validation sets.
        '''
        unique_subjects = np.unique(subject_ids)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        splits = []
        for train_subj_idx, val_subj_idx in kf.split(unique_subjects):
            # Get the subject IDs for the train and validation sets
            train_subjects = unique_subjects[train_subj_idx]
            val_subjects = unique_subjects[val_subj_idx]

            # Extract the indices for the train and validation sets based on the subject IDs
            # train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
            # val_idx   = np.where(np.isin(subject_ids, val_subjects))[0]
            
            # Append each split as a tuple of (train_indices, val_indices)
            splits.append((train_subjects, val_subjects))

        return splits
    
    def train_one_fold(self, model_handler_ins, train_eval_ins, split_ins,
                   X_epoch, train_ids, val_ids,
                   config, device, print_config,
                   single_or_fusion = 'single'):
        
        single_or_fusion = str.lower(single_or_fusion)
        
        if single_or_fusion == 'single':
            # Training set (all except test subject)
            X_train, y_train = split_ins.build_dataset_from_subjects(X_epoch = X_epoch, subjects = train_ids, fs = config['freq'])
            X_val, y_val = split_ins.build_dataset_from_subjects(X_epoch = X_epoch, subjects = val_ids, fs = config['freq'])

            # Build datasets
            train_dataset = SingleManageDataset(X_train, y_train, data_type = config['sensor'])
            val_dataset   = SingleManageDataset(X_val, y_val, data_type = config['sensor'])

            train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size = config["batch_size"], shuffle=False)
        
        elif single_or_fusion == 'fusion':
            EEG_epoch, EMG_epoch = X_epoch
            X_EEG_train, y_EEG_train = split_ins.build_dataset_from_subjects(X_epoch = EEG_epoch, subjects = train_ids, fs = config['eeg_freq'])
            X_EEG_val, y_EEG_val = split_ins.build_dataset_from_subjects(X_epoch = EEG_epoch, subjects = val_ids, fs = config['eeg_freq'])

            X_EMG_train, y_EMG_train = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = train_ids, fs = config['rms_freq'])
            X_EMG_val, y_EMG_val = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = val_ids, fs = config['rms_freq'])

            train_dataset = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)
            val_dataset = MultiManageDataset(X_EEG_val, X_EMG_val, y_EEG_val, y_EMG_val)

            train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size = config["batch_size"], shuffle=False)

        else:
            raise ValueError('Programmer error - Think again dummy :-)') 
        
        model = model_handler_ins.get_model(config = config["model_config"])
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

        best_info = {}
        best_info['val_loss'] = float("inf")
        early_stopping_counter = 0

        for epoch in range(config["epochs"]):

            train_loss = train_eval_ins.train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _ = train_eval_ins.validate_one_epoch(model, val_loader, criterion, device)

            if val_loss < best_info['val_loss']:
                best_info['val_loss'] = val_loss
                best_info['train_loss'] = train_loss
                best_info['val_acc'] = val_acc
                best_info['epoch'] = epoch
                
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config["patience"]:
                    break
            
            print(
                f'Fold {print_config["fold"]} | '
                f'Trial {print_config["trial_id"]}/{print_config["max_num_trials"]} | '
                f'Epoch {epoch+1}/{config["epochs"]} | '
                f'Train {train_loss:.4f} | '
                f'Val {val_loss:.4f} | '
                f'Acc {val_acc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        return best_info
    
    def retrain_model(self, model_handler_ins, train_eval_ins, split_ins,
                   X_epoch, train_subjects_ids,
                   config, mean_epochs, device, print_config,
                   single_or_fusion = 'single'):
        
        if single_or_fusion == 'single':
            # Training set (all except test subject)
            X_train, y_train = split_ins.build_dataset_from_subjects(X_epoch = X_epoch, subjects = train_subjects_ids, fs = config['freq'])

            # Build datasets
            train_dataset = SingleManageDataset(X_train, y_train, data_type = config['sensor'])

            train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle=True)

        elif single_or_fusion == 'fusion':
            EEG_epoch, EMG_epoch = X_epoch
            X_EEG_train, y_EEG_train = split_ins.build_dataset_from_subjects(X_epoch = EEG_epoch, subjects = train_subjects_ids, fs = config['eeg_freq'])
            X_EMG_train, y_EMG_train = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = train_subjects_ids, fs = config['rms_freq'])

            train_dataset = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)

            train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle=True)

        else:
            raise ValueError('Programmer error - Think again dummy :-)') 
        
        # Model
        model = model_handler_ins.get_model(config = config["model_config"])
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

        for epoch in range(mean_epochs):

            train_loss = train_eval_ins.train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            print(
                'Retrain model | '
                f'Trial {print_config["trial_id"]}/{print_config["max_num_trials"]} | '
                f'Epoch {epoch+1}/{mean_epochs} | '
                f'Train {train_loss:.4f} | '
                'Val None | '
                'Acc None |',
                'Early stopping None |',
                end='\r',
                flush=True
            )

        return model, criterion, optimizer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
#========================#
# Dynamic model handling #
#========================#
class SingleNetHandler:
    def __init__(self, model_name : str, sensor_name : str):
        self.model_name = model_name
        self.sensor_name = sensor_name

    def get_hyperparameters(self):
        '''
        Return parameters for a specfic model
        '''

        if self.sensor_name == 'EEG':
            num_hidden_units = [32, 64, 128, 256]
            bidirectional = [False, True]
            lstm_layers = [1, 2, 3]
        else:
            num_hidden_units = [32, 64]
            bidirectional = [False]
            lstm_layers = [1]
        
        common_params = [
            # General
            sherpa.Continuous('learning_rate', [1e-5, 1e-3], scale='log'),      # try out 1e-5, 1e-1
            sherpa.Continuous("weight_decay", [1e-5, 1e-3], scale="log"),       # try out 1e-5, 1e-1
            sherpa.Continuous('dropout', [0.1, 0.4]),
            sherpa.Ordinal('batch_size', [16, 32, 64, 128]),                         
            sherpa.Ordinal('dense_ratio', [0.25, 0.5, 0.75, 1.0]),
            sherpa.Choice('activation', ['relu', 'elu']),
            sherpa.Ordinal('num_hidden_units', num_hidden_units),
            sherpa.Choice("bidirectional", bidirectional),
            sherpa.Choice('lstm_layers', lstm_layers),
        ]

        if self.model_name == "SingleNet_LSTM":
            return common_params
        
        elif self.model_name == "SingleNet_CNN_LSTM" or self.model_name == "SingleNet_CNN_LSTM_ATTENTION":
            cnn_params = [
                sherpa.Ordinal('cnn_filters', [16, 32, 64]),
                sherpa.Ordinal('kernel_ratio', [3, 5, 7, 9, 11]),
            ]
            return common_params + cnn_params
        
        else:
            raise ValueError(f'model_name does not correspond to a model class : {self.model_name}')

    def build_model_config(self, trial : sherpa.Study, input_dim : int, TOTAL_CLASSES : int):
        '''
        Build a config dict for model input
        '''
        config = {
            "input_dim": input_dim,
            "output_dim": TOTAL_CLASSES,
            "hidden_dim": trial.parameters['num_hidden_units'],
            "lstm_layers": trial.parameters['lstm_layers'],
            "bidirectional": trial.parameters['bidirectional'],
            "dropout": trial.parameters['dropout'],
            "activation": trial.parameters['activation'],
            "dense_ratio": trial.parameters['dense_ratio'],
        }

        if self.model_name == "SingleNet_CNN_LSTM" or self.model_name == "SingleNet_CNN_LSTM_ATTENTION":
            config.update({
                "cnn_filters": trial.parameters['cnn_filters'],
                "kernel_size": trial.parameters['kernel_ratio'],
            })

        return config

    def build_training_config(self, trial : sherpa.Study):
        return {
            "lr": trial.parameters["learning_rate"],
            "weight_decay": trial.parameters["weight_decay"],
            "batch_size": trial.parameters["batch_size"],
        }

    def get_model(self, config: dict):
        if self.model_name == "SingleNet_LSTM":
            return SingleNet_LSTM(**config)

        elif self.model_name == "SingleNet_CNN_LSTM":
            return SingleNet_CNN_LSTM(**config)
        
        elif self.model_name == 'SingleNet_CNN_LSTM_ATTENTION':
            return SingleNet_CNN_LSTM_ATTENTION(**config)
    
class FusionNetHandler:
    def __init__(self, model_name : str):
        self.model_name = model_name

    def get_hyperparameters(self):
        '''
        Return parameters for a specfic model
        '''

        common_params = [
            # General
            sherpa.Continuous('learning_rate', [1e-5, 1e-3], scale='log'),      # try out 1e-5, 1e-1
            sherpa.Continuous("weight_decay", [1e-5, 1e-3], scale="log"),       # try out 1e-5, 1e-1
            sherpa.Continuous('dropout', [0.1, 0.4]),
            sherpa.Ordinal('batch_size', [16, 32, 64, 128]),
            sherpa.Ordinal('dense_ratio', [0.25, 0.5, 0.75, 1.0]),
            sherpa.Choice('activation', ['relu', 'elu']),
            # LSTM
            sherpa.Ordinal('num_hidden_units', [32, 64, 128, 256]),
            sherpa.Choice("bidirectional", [False, True]),
            sherpa.Choice('lstm_layers', [1, 2, 3]),
        ]

        if self.model_name == "FusionNet_LSTM":
            return common_params

        elif self.model_name == "FusionNet_CNN_LSTM" or self.model_name == "FusionNet_CNN_LSTM_ATTENTION":
            cnn_params = [
                sherpa.Ordinal('cnn_filters', [16, 32, 64]),
                sherpa.Ordinal('kernel_ratio', [3, 5, 7, 9, 11]),
            ]
            return common_params + cnn_params
        else:
            raise ValueError(f'model_name does not correspond to a model class : {self.model_name}')
        
    def build_model_config(self, trial : sherpa.Study, EEG_CH : int, EMG_CH : int, EEG_CLASSES : int, EMG_CLASSES : int, TOTAL_CLASSES : int):
        '''
        Build a config dict for model input
        '''
        config = {
            "eeg_dim": EEG_CH,
            "emg_dim": EMG_CH,
            "eeg_output_dim": EEG_CLASSES,
            "emg_output_dim": EMG_CLASSES,
            "output_dim": TOTAL_CLASSES,
            "hidden_dim": trial.parameters['num_hidden_units'],
            "lstm_layers": trial.parameters['lstm_layers'],
            "bidirectional": trial.parameters['bidirectional'],
            "dropout": trial.parameters['dropout'],
            "activation": trial.parameters['activation'],
            "dense_ratio": trial.parameters['dense_ratio'],
        }

        if self.model_name == "FusionNet_CNN_LSTM" or self.model_name == "FusionNet_CNN_LSTM_ATTENTION":
            config.update({
                "cnn_filters": trial.parameters['cnn_filters'],
                "kernel_size": trial.parameters['kernel_ratio'],
            })

        return config
    
    def build_training_config(self, trial : sherpa.Study):
        return {
            "lr": trial.parameters["learning_rate"],
            "weight_decay": trial.parameters["weight_decay"],
            "batch_size": trial.parameters["batch_size"],
        }
    
    def get_model(self, config: dict):
        if self.model_name == "FusionNet_LSTM":
            return FusionNet_LSTM(**config)

        elif self.model_name == "FusionNet_CNN_LSTM":
            return FusionNet_CNN_LSTM(**config)
        
        elif self.model_name == "FusionNet_CNN_LSTM_ATTENTION":
            return FusionNet_CNN_LSTM_ATTENTION(**config)

class ExperimentLogger:

    def __init__(self, save_path : Path):
        self.save_path = save_path / 'SHERPA_results.pt'
        
        # Load existing file if present
        if os.path.exists(self.save_path):
            self.results = torch.load(self.save_path)
        else:
            self.results = {"trials": []}

    def log_trial(
        self,
        trial_id,
        hyperparams,
        best_epoch,
        train_loss,
        val_loss,
        val_acc,
        test_loss,
        test_acc,
        preds,
        labels):
        """
        Append one trial result.
        """

        trial_result = {
            "trial_id": trial_id,
            "hyperparameters": hyperparams,
            "best_epoch": best_epoch,
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "predictions": preds,
            "labels": labels
        }

        self.results["trials"].append(trial_result)

        # Save immediately (safe against crashes)
        torch.save(self.results, self.save_path)

#==========================#
# Real time classification #
#==========================#
class BandpassFilter:
    def __init__(self, fs, lowcut, highcut, order=4):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        nyq = fs / 2
        low = lowcut / nyq
        high = highcut / nyq

        self.b, self.a = butter(order, [low, high], btype='band')
        self.zi = None

    def update(self, x):
        """
        x: shape (N, channels)
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.zi is None:
            zi_base = lfilter_zi(self.b, self.a)
            self.zi = np.tile(zi_base[:, None], (1, x.shape[1])) * x[0]

        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y

class NotchFilter:
    def __init__(self, fs, cutoff=50, Q=30):
        self.fs = fs
        self.cutoff = cutoff
        self.Q = Q

        # Design filter
        w0 = cutoff / (fs / 2)
        self.b, self.a = iirnotch(w0, Q)

        self.zi = None  # filter state

    def update(self, x):
        """
        x: shape (N, channels)
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.zi is None:
            # initialize per channel
            self.zi = np.tile(lfilter_zi(self.b, self.a), (x.shape[1], 1)).T

        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y

class LowpassFilter:
    def __init__(self, fs, cutoff, order=2):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order

        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        self.b, self.a = butter(order, normal_cutoff, btype='low')
        self.zi = None

    def update(self, x):
        """
        x: shape (N, channels)
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.zi is None:
            # steady-state init
            zi_base = lfilter_zi(self.b, self.a)
            self.zi = np.tile(zi_base[:, None], (1, x.shape[1])) * x[0]

        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y

class EMANormalizer:
    def __init__(self, alpha=0.999, eps=1e-8):
        self.alpha = alpha
        self.eps = eps

        self.mu = None
        self.var = None

    def update(self, x):
        """
        x: shape (N, channels)
        returns normalized x
        """
        x = np.asarray(x, dtype=np.float64)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        if self.mu is None or self.var is None:
            self.mu = batch_mean
            self.var = batch_var

        x_norm = (x - self.mu) / (np.sqrt(self.var) + self.eps)

        # Update EMA estimates
        self.mu = self.alpha * self.mu + (1 - self.alpha) * batch_mean
        self.var = self.alpha * self.var + (1 - self.alpha) * batch_var

        return x_norm
    
    def train_EMA_coefficients(self, data : np.ndarray, sliding_window_size : int = 1000):
        if self.mu is not None or self.var is not None:
            raise ValueError("EMA coefficients have already been trained. Re-instantiate EMGStreamProcessor to reset.")
        # Train EMA coefficients by running through the data once
        for trial_start in range(0, data.shape[0], sliding_window_size):
            trial_end = trial_start + sliding_window_size
            segment = data[trial_start:trial_end]
            self.update(segment)

        print('EMA : Initilized mean:', self.mu)
        print('EMA : Initilized variance:', self.var)

class EMGStreamProcessor:
    def __init__(self, fs, lowcut, highcut,
                 reject_config_dict : Dict,
                 rms_window=200, rms_step=50,
                 hampel_window=200, hampel_sigma=3,
                 base_dir : str = 'to_data_folder',
                 disable_rejection = False):

        self.fs = fs
        self.notch_ins = NotchFilter(fs = fs, cutoff = 50, Q = 30)
        self.bandpass_ins = BandpassFilter(fs = fs, lowcut = lowcut, highcut = highcut, order = 4)
        self.lowpass_ins = LowpassFilter(fs = fs, cutoff = 5, order = 2)
        # self.ema_ins = EMANormalizer(alpha=0.999, eps=1e-8)

        self.rms_window = rms_window
        self.rms_step = rms_step
        self.hampel_window = hampel_window
        self.hampel_sigma = hampel_sigma
        self.first_update = False
        
        self.base_dir = base_dir
        self.reject_config_dict = reject_config_dict
        self.disable_rejection = disable_rejection
    
    def reset(self):
        self.notch_ins = NotchFilter(fs = self.fs, cutoff = 50, Q = 30)
        self.bandpass_ins = BandpassFilter(fs = self.fs, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT, order = 4)
        self.ema_ins = EMANormalizer(alpha=0.999, eps=1e-8)
        self.lowpass_ins = LowpassFilter(fs = self.fs, cutoff = 5, order = 2)
        self.first_update = False

    def update(self, chunk):
        """
        Preprocess a chunk of EMG data through the pipeline:
        1) Notch filter (50 Hz)
        2) Bandpass filter (20-450 Hz)
        3) Hampel filter (window size = 100, n_sigmas = 2.0)
        4) RMS (window size = 200, step size = 50)
        5) EMA normalization
        """

        #===========================#
        # 1) Filtering (continuous) #
        #===========================#
        notch = self.notch_ins.update(chunk)
        bandpass = self.bandpass_ins.update(notch)

        if not self.first_update:               # Don't append first update to avoid filter transients
            self.first_update = True
            return None

        #==================#
        # 2) Hampel filter #
        #==================#
        hampel = self._hampel(bandpass, window_size = self.hampel_window, n_sigmas = self.hampel_sigma)

        #==================#
        # 3) RMS (sliding) #
        #==================#
        rms = self._rms_causal(hampel, window_size = self.rms_window, step_size = self.rms_step)

        #==================#
        # 4) Normalization #
        #==================#
        # norm = self.ema_ins.update(rms)

        #==================#
        # 5) Low pass filt #
        #==================#
        lowpass = self.lowpass_ins.update(rms)

        return lowpass

    def load_subject_data(self, subj : str, finger : str, modality : str, trim_period : int = 3, trial_period : int = 9):
        '''
        This method is only used for loading data for real-time classification.
        It loads, trims the edges and move bad epochs
        '''
        load_ins = load_datasets(base_dir = self.base_dir)
        reject_ins = RejectBadEpochs(base_dir = self.base_dir)

        file_path = load_ins.find_flex_files(subjects = subj,
                                             modality = modality,
                                             fingers = finger,
                                             prefix = 'flex'
                                             )
        
        data_container = []
        epochs_overview = []

        for file in file_path:
            print(file)
            data = pd.read_csv(file).to_numpy()

            #============#
            # Trim edges #
            #============#
            trim_samples = self.fs * trim_period
            samples_per_epoch = self.fs * trial_period

            valid_samples = data.shape[0] - 2 * trim_samples            # Total samples for experimental period. WHY *2 : Trim egde on both sides
            num_epochs = int( np.round(valid_samples / samples_per_epoch) )     # Divide out total samples in sections of samples per epoch -> Results in number of epochs

            trim_start = trim_samples
            trim_end = trim_start + num_epochs * samples_per_epoch              # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen

            data_trim = data[trim_start : trim_end, :]

            data_container.append(data_trim)
            epochs_overview.append(num_epochs)
            
        data_container = np.concatenate(data_container, axis=0)

        #===================#
        # Reject bad epochs #
        #===================#
        print('DEMO PURPOSE - FUNCTIONS DISABLE')
        if not self.disable_rejection:
            reject_mask = reject_ins.reject_routine(data_file_per_finger = file_path,
                                                epochs_overview = epochs_overview,
                                                EEG_data = None,
                                                RMS_data = data_container,
                                                reject_config_dict = self.reject_config_dict,
                                                EEG_useable_channels = None)

        total_epochs = sum(epochs_overview)
        # EMG_epoch = data_container.reshape(total_epochs, data_container.shape[0] // total_epochs, data_container.shape[1])

        # EMG_epoch_clean = EMG_epoch[~reject_mask]

        # total_clean_epochs = EMG_epoch_clean.shape[0]
        EMG_epoch_clean = data_container
        total_clean_epochs = 0
        return EMG_epoch_clean, total_clean_epochs
    
    def relabel_windows(self, epochs : np.ndarray, window_samples : int = 1000, step_samples : int = 200, fs : int = EMG_FREQ, labels : list = ["rest", "contract", "release"]):
        '''
        This method takes the epochs and converts to continuous data and relabels it according to the experimental protocol.
        It produces a window of 1000 samples and a step size of 200 samples.
        Each period will contain 26 windows. 1 trial = 78 windows

        Returns
        -------
        filtered_epochs : np.ndarray
            Shape (n_windows, n_samples, n_channels)
            Each window is trial*periods*steps (26 steps per period)
        '''
        WINDOW_SAMPLES = window_samples
        STEP_SAMPLES = step_samples

        TRIAL_SAMPLES = 9 * fs
        SEGMENT_SAMPLES = 3 * fs

        labels = labels

        window_labels = []
        filtered_epochs = []

        data = epochs.reshape(-1, epochs.shape[-1])
        n_samples = data.shape[0]

        # self.reset()        # For every subject reset the filters and EMA normalizer to avoid data leakage between subjects

        # Loop over trials in continuous data
        for trial_start in range(0, n_samples, TRIAL_SAMPLES):

            trial_end = trial_start + TRIAL_SAMPLES
            if trial_end > n_samples:
                break

            # Loop over segments inside trial
            for seg_idx in range(3):
                seg_start = trial_start + seg_idx * SEGMENT_SAMPLES
                seg_end   = seg_start + SEGMENT_SAMPLES

                segment = data[seg_start:seg_end]

                # Sliding window inside segment
                for start in range(0, SEGMENT_SAMPLES - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                    end = start + WINDOW_SAMPLES

                    chunk = segment[start:end]

                    # print(f'start {start} to {end}')

                    filtered_data = self.update(chunk)

                    if filtered_data is None:
                        continue
                    
                    REST_THRESH = 0.003

                    # Combine channels
                    envelope = np.mean(filtered_data, axis=1)

                    # Activity magnitude
                    activity = np.mean(envelope)

                    # Rising/falling behavior
                    slope = np.mean(np.diff(envelope))
                    
                    if seg_idx == 1:
                        pass
                    
                    if activity < REST_THRESH:
                        label = 'rest'
                    elif slope > 0:
                        label = 'contract'
                    else:
                        label = 'release'
                    '''
                    # Ensure 1sec of contract and the last 2 sec of release are labeled rest
                    if seg_idx == 1 and end <= fs:     # Until 1 sec of contract will be labeled rest
                        label = "rest"
                    elif seg_idx == 2 and start > fs:  # Beyond 1 sec of release will be labeled rest
                        label = "rest"
                    else:
                        label = labels[seg_idx]
                    '''
                    window_labels.append(label)
                    filtered_epochs.append(filtered_data)
        
        return np.array(filtered_epochs), np.array(window_labels)

    def _hampel(self, x: np.ndarray, window_size: int = 100, n_sigmas: float = 3.0, plot_filter_results: list = [False, None]):
        """
        Hampel filter for multi-channel signals.

        Parameters
        ----------
        x : np.ndarray
            Shape (n_samples, n_channels)
        window_size : int
            Number of samples on EACH side of the center sample
        n_sigmas : float
            Threshold multiplier
        plot_filter_results : list
            First element -> bool to display filter results
            second element -> list of specfic time window or None to display all

        Returns
        -------
        filtered_data   : np.ndarray (n_samples, n_channels)
        """

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("Input must be 2D: (samples, channels)")

        # Scale factor to make MAD comparable to standard deviation
        # (valid for approximately Gaussian data)
        k_scale = 1.4826
        kernel = 2 * window_size + 1

        # Median per channel (filter only along time axis)
        medians = median_filter(
            x,
            size=(kernel, 1),
            mode="reflect"
        )

        # Difference between real signal and the typical values of the signal
        diff = np.abs(x - medians)

        # Robust estimate of local variability (Median Absolute Deviation) per channel
        mad = k_scale * median_filter(
            diff,
            size=(kernel, 1),
            mode="reflect"
        )

        thresholds = n_sigmas * mad

        outlier_mask = diff > thresholds

        x_filt = x.copy()
        x_filt[outlier_mask] = medians[outlier_mask]

        if plot_filter_results[0]:
            # Collect outlier indices per channel
            outlier_indices = [
                np.nonzero(outlier_mask[:, ch])[0].tolist()
                for ch in range(x.shape[1])
            ]
            for ch in range(x.shape[1]):
                print('Number of outliers:', len(outlier_indices[ch])) 

            self.plot_hampel_filter(original_signal = x, filtered_signal = x_filt, outlier_indices = outlier_indices, medians = medians, thresholds = thresholds, zoom = plot_filter_results[1])
        
        return x_filt

    def _rms_conv(self, signal, window_size=200, step_size=25):
        '''
        Convolution RMS. RMS = sqrt( LPF(x^2) ), where LPF is implemented as a uniform filter (moving average) over the squared signal.
        '''
        power = signal**2

        mean_power = uniform_filter1d(
            power,
            size=window_size,
            axis=0,
            mode="nearest"
        )

        rms = np.sqrt(mean_power)

        return rms[::step_size]
    
    def _rms_causal(self, signal, window_size=200, step_size=50):
        signal = np.asarray(signal, dtype=float)
        power = signal ** 2
        out = []

        for end in range(window_size, len(signal) + 1, step_size):
            start = end - window_size
            win = power[start:end]
            out.append(np.sqrt(np.mean(win, axis=0)))

        return np.array(out)
#=================#
# Quick functions #
#=================#
def check_model(model_name : str = None, sensor_name : str = None, num_motions : int = None):
    sn = str.upper(sensor_name) if sensor_name is not None else None
    mn = model_name

    network = mn.split('_')[0]

    if network == 'FusionNet':
        if mn != 'FusionNet_LSTM' and mn != 'FusionNet_CNN_LSTM' and mn != 'FusionNet_CNN_LSTM_ATTENTION': 
            raise ValueError(f'model_name : {mn} not valid')
    else:
        if sn != 'EEG' and sn != 'EMG':
            raise ValueError(f'sensor_name : {sn} not valid')
        elif mn != 'SingleNet_LSTM' and mn != 'SingleNet_CNN_LSTM' and mn != 'SingleNet_CNN_LSTM_ATTENTION':
            raise ValueError(f'model_name : {mn} not valid')
        
    if num_motions != 2 and num_motions != 7 and num_motions != 3:
        raise ValueError(f'Num motions : {num_motions} not valid') 

def compute_class_weights(y_train):

    classes, counts = np.unique(y_train, return_counts=True)

    print("Class distribution:")
    for c, cnt in zip(classes, counts):
        print(f"Class {c}: {cnt}")

    # Inverse frequency weighting
    weights = 1.0 / counts

    # Normalize (optional but recommended)
    weights = weights / weights.sum() * len(classes)

    class_weights = np.zeros(len(classes))
    class_weights[classes] = weights

    print("Class weights:", class_weights)
    return class_weights

def normalize_global_per_channel(X_epoch, train_subject_ids):
    """
    Compute global mean/std from TRAIN data only (no leakage),
    and apply to train/val/test for all subjects and modalities.

    Parameters
    ----------
    X_epoch : dict
        Structure: X_epoch[subj][ml][split] -> (samples, seq_len, channels)

    Returns
    -------
    X_epoch : dict
        Normalized data (in-place modified)
    mu : np.ndarray
        Mean per channel (shape: channels,)
    sigma : np.ndarray
        Std per channel (shape: channels,)
    """

    # =========================
    # 1) Collect all TRAIN data
    # =========================
    X_train_all = []

    for subj in train_subject_ids:
        for ml in X_epoch[subj]:
            X_train_all.append(X_epoch[subj][ml]['train'])

    X_train_all = np.concatenate(X_train_all, axis=0)  # (N, seq_len, ch)

    # =========================
    # 2) Compute mean/std
    # =========================
    mu = np.mean(X_train_all, axis=(0, 1))       # (channels,)
    sigma = np.std(X_train_all, axis=(0, 1))     # (channels,)
    # mu = np.mean(X_train_all)       # global
    # sigma = np.std(X_train_all)     # global

    # =========================
    # 3) Normalize all splits
    # =========================
    norm_epochs = {}
    eps = 1e-8
    for subj in X_epoch:
        norm_epochs[subj] = {}

        for ml in X_epoch[subj]:
            norm_epochs[subj][ml] = {}

            for split in ['train', 'val', 'test']:
                X = X_epoch[subj][ml][split]

                X_norm = (X - mu) / (sigma + eps)

                norm_epochs[subj][ml][split] = X_norm

    return norm_epochs, mu, sigma
#==============================#
# Traning of model Per subject #
#==============================#

def singleNet_classfication_real_time(subject_name : str | list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', sensor_name : str = None, model_name : str = None, num_motions : int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    sensor_name = str.upper(sensor_name)
    FREQ = EMG_FREQ if sensor_name == 'EMG' else EEG_FREQ

    check_model(model_name = model_name, sensor_name = sensor_name, num_motions = num_motions)

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    split_ins = Manage3Split(seed = SEED)
    model_handler_ins = SingleNetHandler(model_name = model_name, sensor_name = sensor_name)
    EMG_ins = EMGStreamProcessor(fs = FREQ, lowcut=EMG_LOWCUT, highcut=EMG_HIGHCUT,
                             reject_config_dict = REJECT_CONFIG_DICT,
                             rms_window = 500, rms_step = 50,
                             hampel_window = 100, hampel_sigma = 2, base_dir = data_dir)
    if sensor_name == 'EMG':
        pass
    else:
        raise ValueError('Can currently only be EMG modaility')
    #===========#
    # Load data #
    #===========#
    X_epoch = {}
    X_labels = {}
    if num_motions == 7:
        motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip']
    elif num_motions == 3:
        motion_list = ['index', 'thumb', 'pinchGrip']
    elif num_motions == 2:
        motion_list = ['index', 'thumb']
    SUBJECTS_IDs = ['subject_0']
    
    for subj in SUBJECTS_IDs:
        X_epoch[subj] = {}
        X_labels[subj] = {}

        for ml in motion_list:
            data, num_epochs = EMG_ins.load_subject_data(subj = subj, finger = ml, modality = sensor_name, trim_period = TRIM_PERIOD, trial_period = TRIAL_PERIOD)    

            # Trial-level split
            train_idx, val_idx, test_idx = split_ins._split_trials(num_trials = num_epochs, train_ratio = 0.7)

            # Split each window into train, val, test sets
            X_epoch[subj][ml] = {}
            X_labels[subj][ml] = {}

            for split_name, split_indicies in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
                split_data = data[split_indicies]
                
                X_e, y_e = EMG_ins.relabel_windows(
                    epochs = split_data,
                    window_samples = 1000,
                    step_samples = 200,
                    fs = EMG_FREQ,
                    labels = ['rest', 'contract', 'release']
                )

                X_epoch[subj][ml][split_name] = X_e
                X_labels[subj][ml][split_name] = y_e

    X_filt, mu, sigma = normalize_global_per_channel(X_epoch, train_subject_ids = SUBJECTS_IDs)
    np.save(f"{log_dir}/mu.npy", mu)
    np.save(f"{log_dir}/sigma.npy", sigma)
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        X, y = split_ins.build_dataset_window_relabel(
            X_epoch = X_filt,
            X_labels = X_labels,
            subjects = SUBJECTS_IDs,
            split = split
        )      

        datasets[split] = (X, y) 
    
    X_train, y_train = datasets['train'] 
    X_val, y_val = datasets['val'] 
    X_test, y_test = datasets['test'] 

    _, _, num_channels = X_train.shape

    #=================#
    # Single datasets #
    #=================#
    train_eval_ins = SingleNet_train_eval()

    print('\nTraining dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train, data_type = sensor_name)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val, data_type = sensor_name)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = sensor_name)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 1             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 15
    DATA_CH = num_channels
    NUM_CLASSES = (2 * num_motions + 1) if sensor_name == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25 
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    # algorithm = sherpa.algorithms.GPyOpt(
    #     max_num_trials = MAX_NUM_TRIALS,
    #     acquisition_type = 'EI',                     # Expected improvement
    #     num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    # )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    # Class weights (Unbalanced classes)
    # class_weights = compute_class_weights(y_train = y_train)
    # class_weights = torch.tensor(class_weights, dtype = torch.float32).to(device)

    for trial in study:
        # model_config = model_handler_ins.build_model_config(
        #     trial = trial,
        #     input_dim = DATA_CH,
        #     TOTAL_CLASSES = NUM_CLASSES
        # )
        ''' 3 motion
        model_config = {
            "input_dim": 3,
            "output_dim": 7,
            "hidden_dim": 128,
            "lstm_layers": 2,
            "bidirectional": False,
            "dropout": 0.14853567226828412,
            "activation": 'elu',
            "dense_ratio": 0.25,
            "cnn_filters": 64,
            "kernel_size": 9,
        }
        train_config = {
            "lr": 3.4621999436698846e-05,
            "weight_decay": 0.0008727684532800766,
            "batch_size": 128,
        }
        '''
        model_config = {
            "input_dim": 3,
            "output_dim": 15,
            "hidden_dim": 256,
            "lstm_layers": 3,
            "bidirectional": True,
            "dropout": 0.3118194976337473,
            "activation": 'relu',
            "dense_ratio": 0.75,
            "cnn_filters": 32,
            "kernel_size": 7,
        }
        train_config = {
            "lr": 0.00015095576430059946,
            "weight_decay": 0.0006720891424495656,
            "batch_size": 128,
        }
        #=================#
        # Single datasets #
        #=================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validate_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)
        
        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss
            print(f'Global best vloss: {best_val_loss}. Test accuracy: {test_acc}. Val accuracy: {best_val_acc}',)

            torch.save({
                "model_name": model_name,
                "sensor_name": sensor_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def singleNet_classfication_dependent(subject_name : str | list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', sensor_name : str = None, model_name : str = None, num_motions : int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    sensor_name = str.upper(sensor_name)

    check_model(model_name = model_name, sensor_name = sensor_name, num_motions = num_motions)

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    model_handler_ins = SingleNetHandler(model_name = model_name, sensor_name = sensor_name)
    
    if sensor_name == 'EMG':
        EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    elif sensor_name == 'EEG':
        EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    else:
        raise ValueError('sensor_name must be either EMG or EEG')

    #===========#
    # Load data #
    #===========#
    X_epoch = {}
    motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip'] if num_motions == 7 else ['index', 'thumb']

    for ml in motion_list:
        if sensor_name == 'EMG':
            X_epoch[ml], _, _ = load_ins.load_EMG_data(subject_name = subject_name, finger_name = ml, EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
        else:
            X_epoch[ml], _ = load_ins.load_EEG_data(subject_name = subject_name, finger_name = ml, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    FREQ = RMS_FREQ if sensor_name == 'EMG' else EEG_FREQ

    X_train, X_val, X_test, y_train, y_val, y_test = split_ins.build_modality_split(
        epoch_dict = X_epoch,
        fs = FREQ
    )
    
    _, _, num_channels = X_train.shape

    #=================#
    # Single datasets #
    #=================#
    train_eval_ins = SingleNet_train_eval()

    print('\nTraining dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train, data_type = sensor_name)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val, data_type = sensor_name)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = sensor_name)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 75
    DATA_CH = num_channels
    NUM_CLASSES = (2 * num_motions + 1) if sensor_name == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25 
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial = trial,
            input_dim = DATA_CH,
            TOTAL_CLASSES = NUM_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )

        #=================#
        # Single datasets #
        #=================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validate_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)
        
        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss

            torch.save({
                "model_name": model_name,
                "sensor_name": sensor_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def fusionNet_classfication_dependent(subject_name : str | list, sherpa_log_folder : str = 'fusionNet_LSTM', model_name : str = 'FusionNet_LSTM'):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = f'{subject_name}'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)    

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    train_eval_ins = FusionNet_train_eval()
    model_handler_ins = FusionNetHandler(model_name = model_name)
    #===========#
    # Load data #
    #===========#
    EEG_epoch_index, EMG_epoch_index, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
    EEG_epoch_thumb, EMG_epoch_thumb, _, _ = load_ins.load_EEG_EMG_data(subject_name = subject_name, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)

    num_index_trials = EEG_epoch_index.shape[0]
    num_thumb_trials = EEG_epoch_thumb.shape[0]

    X_EEG_train, X_EEG_val, X_EEG_test, y_EEG_train, y_EEG_val, y_EEG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EEG_epoch_index,
        epoch_thumb = EEG_epoch_thumb,
        fs = EEG_FREQ
    )

    X_EMG_train, X_EMG_val, X_EMG_test, y_EMG_train, y_EMG_val, y_EMG_test = split_ins.build_modality_split(
        num_index_trials = num_index_trials,
        num_thumb_trials = num_thumb_trials,
        epoch_index = EMG_epoch_index,
        epoch_thumb = EMG_epoch_thumb,
        fs = RMS_FREQ
    )

    _, EEG_num_samples, EEG_num_channels = X_EEG_train.shape
    _, EMG_num_samples, EMG_num_channels = X_EMG_train.shape

    #=======================#
    # Multi fusion datasets #
    #=======================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)
    print('Validation dataset shapes:')
    val_dataset_ins = MultiManageDataset(X_EEG_val, X_EMG_val, y_EEG_val, y_EMG_val)
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 20
    EEG_CH = EEG_num_channels
    EMG_CH = EMG_num_channels
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25

    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial,
            EEG_CH, EMG_CH,
            EEG_CLASSES, EMG_CLASSES,
            TOTAL_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )
        #=======================#
        # Multi fusion datasets #
        #=======================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===============================================================#
        # NOTE: Tensorboard config                                      #
        #   Enable all if using tensorboard                             #
        #   Change log_dir -> log_folder when saving model : torch.save #
        #===============================================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validate_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)
            
            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'{subject_name} | '
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)

        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss
            
            torch.save({
                "model_name": model_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))

        # writer.close()
        study.finalize(trial, status = 'COMPLETED')

#==================================#
# Traning of model across subjects #
#==================================#
def singleNet_Kfold_classfication_independent(sherpa_log_folder : str = 'subject_dependent/SingleNet_LSTM_EMG', sensor_name : str = None, model_name : str = 'SingleNet_LSTM', num_motions : int = 2):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    check_model(model_name = model_name, sensor_name = sensor_name, num_motions = num_motions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'all_subjects'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    sensor_name = str.upper(sensor_name)

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    train_eval_ins = SingleNet_train_eval()
    model_handler_ins = SingleNetHandler(model_name = model_name, sensor_name = sensor_name)
    Kfold_ins = KFoldManageDataset()

    if sensor_name == 'EMG':
        EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    elif sensor_name == 'EEG':
        EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    else:
        raise ValueError('data_type must be either EMG or EEG')

    #====================#
    # Load Training data #
    #====================#
    SUBJECT_IDs = [f'subject_{i}' for i in range(0, 17)]
    TEST_SUBJECT = ['subject_8']    
    X_epoch = {}

    motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip'] if num_motions == 7 else ['index', 'thumb']
    for subj in SUBJECT_IDs:
        X_epoch[subj] = {}

        for ml in motion_list:
            if sensor_name == 'EMG':
                X_epoch[subj][ml], _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = ml, EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
            else:
                X_epoch[subj][ml], _ = load_ins.load_EEG_data(subject_name = subj, finger_name = ml, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)
    
    # Provide a list of subjects to split between train and val. Exclude the test subject from this list.
    train_subjects_ids = []
    for subj in SUBJECT_IDs:
        if subj not in TEST_SUBJECT:
            train_subjects_ids.append(subj)

    
    FREQ = RMS_FREQ if sensor_name == 'EMG' else EEG_FREQ    

    # Prepare test dataset
    X_test, y_test = split_ins.build_dataset_from_subjects(X_epoch = X_epoch, subjects = TEST_SUBJECT, fs = FREQ)
    test_dataset  = SingleManageDataset(X_test, y_test, data_type = sensor_name)
    
    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 25             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 15
    DATA_CH = EMG_NUM_CH if sensor_name == 'EMG' else EEG_NUM_CH
    NUM_CLASSES = 5 if sensor_name == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    splits = Kfold_ins.create_kfold_splits_subject_independent(subject_ids = train_subjects_ids, k = 8)

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial = trial,
            input_dim = DATA_CH,
            TOTAL_CLASSES = NUM_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )

        FOLD_INFO = []

        config = {
            "model_config" : model_config,
            "lr": train_config["lr"],
            "weight_decay": train_config["weight_decay"],
            "batch_size": train_config["batch_size"],
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "sensor": sensor_name,
            "freq" : FREQ,
            }

        for fold, (train_ids, val_ids) in enumerate(splits):

            print_config = {
                'trial_id': trial.id,
                'max_num_trials': MAX_NUM_TRIALS,
                'fold': fold,
            }

            best_info = Kfold_ins.train_one_fold(
                        model_handler_ins = model_handler_ins,
                        train_eval_ins = train_eval_ins,
                        split_ins = split_ins,
                        X_epoch = X_epoch,
                        train_ids = train_ids,
                        val_ids = val_ids,
                        config = config, 
                        device = device,
                        print_config = print_config)

            FOLD_INFO.append(best_info)
        
        fold_val_losses = [f["val_loss"] for f in FOLD_INFO]
        fold_val_accs   = [f["val_acc"] for f in FOLD_INFO]
        fold_train_losses = [f["train_loss"] for f in FOLD_INFO]
        fold_epochs     = [f["epoch"] for f in FOLD_INFO]

        avg_fold_vloss = np.mean(fold_val_losses)
        avg_fold_epochs = int(np.round(np.mean(fold_epochs)))

        # Load new model
        # Train on all data
        # Extract the model to do inference
        retrain_model, retrain_criterion, retrain_optimizer = Kfold_ins.retrain_model(
            model_handler_ins = model_handler_ins,
            train_eval_ins = train_eval_ins,
            split_ins = split_ins,
            X_epoch = X_epoch,
            train_subjects_ids = train_subjects_ids,
            config = config,
            mean_epochs = avg_fold_epochs,
            device = device,
            print_config = print_config
        )
        
        # Prepara test dataset for inference
        test_loader  = DataLoader(test_dataset, batch_size = config["batch_size"], shuffle=False)
        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = retrain_model, test_loader = test_loader, criterion = retrain_criterion, device = device)

        study.add_observation(
            trial = trial,
            objective = avg_fold_vloss,
            iteration = 0
        )

        study.finalize(trial)

        if avg_fold_vloss < global_best_vloss :
            global_best_vloss = avg_fold_vloss
            
            torch.save({
                "model_name": model_name,
                "sensor_name": sensor_name,
                "model_state": retrain_model.state_dict(),
                "model_args": model_config, 
                "optimizer_state_dict": retrain_optimizer.state_dict(),
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))


        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = fold_epochs,
            train_loss = fold_train_losses,
            val_loss = fold_val_losses,
            val_acc = fold_val_accs,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def fusionNet_Kfold_classfication_independent(sherpa_log_folder : str = 'subject_dependent/SingleNet_LSTM_EMG', model_name : str = 'SingleNet_LSTM', num_motions = 2):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    check_model(model_name = model_name, sensor_name = None, num_motions = num_motions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'all_subjects'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    train_eval_ins = FusionNet_train_eval()
    model_handler_ins = FusionNetHandler(model_name = model_name)
    Kfold_ins = KFoldManageDataset()

    #====================#
    # Load Training data #
    #====================#
    SUBJECT_IDs = [f'subject_{i}' for i in range(0, 17)]
    TEST_SUBJECT = ['subject_8']    
    EEG_epoch = {}
    EMG_epoch = {}

    motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip'] if num_motions == 7 else ['index', 'thumb']
    for subj in SUBJECT_IDs:
        EEG_epoch[subj] = {}
        EMG_epoch[subj] = {}

        for ml in motion_list:
            print(f'Load for {subj} for {ml}')
            eeg_temp, emg_temp, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj,
                                                                    finger_name = ml,
                                                                    reject_config_dict = REJECT_CONFIG_DICT,
                                                                    EEG_preprocessing_func = EEG_ins.preprocessing_routine,
                                                                    EMG_preprocessing_func = EMG_ins.preprocessing_routine,
                                                                    EMG_config_dict = EMG_CONFIG_DICT,
                                                                    EEG_useable_channels = EEG_USEABLE_CHANNELS)
            EEG_epoch[subj][ml] = eeg_temp
            EMG_epoch[subj][ml] = emg_temp
    
    # Provide a list of subjects to split between train and val. Exclude the test subject from this list.
    train_subjects_ids = []
    for subj in SUBJECT_IDs:
        if subj not in TEST_SUBJECT:
            train_subjects_ids.append(subj)

    # Prepare test dataset
    X_EEG_test, y_EEG_test = split_ins.build_dataset_from_subjects(X_epoch = EEG_epoch, subjects = TEST_SUBJECT, fs = EEG_FREQ)
    X_EMG_test, y_EMG_test = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = TEST_SUBJECT, fs = RMS_FREQ)
    
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)
    
    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 50             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 30
    EEG_CH = X_EEG_test.shape[2]
    EMG_CH = X_EMG_test.shape[2]
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    K_folds = 8
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    splits = Kfold_ins.create_kfold_splits_subject_independent(subject_ids = train_subjects_ids, k = K_folds)

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial,
            EEG_CH, EMG_CH,
            EEG_CLASSES, EMG_CLASSES,
            TOTAL_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )

        FOLD_INFO = []

        config = {
            "model_config" : model_config,
            "lr": train_config["lr"],
            "weight_decay": train_config["weight_decay"],
            "batch_size": train_config["batch_size"],
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "eeg_freq" : EEG_FREQ,
            "rms_freq" : RMS_FREQ
            }

        for fold, (train_ids, val_ids) in enumerate(splits):

            print_config = {
                'trial_id': trial.id,
                'max_num_trials': MAX_NUM_TRIALS,
                'fold': fold,
            }

            best_info = Kfold_ins.train_one_fold(
                        model_handler_ins = model_handler_ins,
                        train_eval_ins = train_eval_ins,
                        split_ins = split_ins,
                        X_epoch = (EEG_epoch, EMG_epoch),
                        train_ids = train_ids,
                        val_ids = val_ids,
                        config = config, 
                        device = device,
                        print_config = print_config,
                        single_or_fusion = 'fusion')

            FOLD_INFO.append(best_info)
        
        fold_val_losses = [f["val_loss"] for f in FOLD_INFO]
        fold_val_accs   = [f["val_acc"] for f in FOLD_INFO]
        fold_train_losses = [f["train_loss"] for f in FOLD_INFO]
        fold_epochs     = [f["epoch"] for f in FOLD_INFO]

        avg_fold_vloss = np.mean(fold_val_losses)
        avg_fold_epochs = int(np.round(np.mean(fold_epochs)))

        # Load new model
        # Train on all data
        # Extract the model to do inference
        retrain_model, retrain_criterion, retrain_optimizer = Kfold_ins.retrain_model(
            model_handler_ins = model_handler_ins,
            train_eval_ins = train_eval_ins,
            split_ins = split_ins,
            X_epoch = (EEG_epoch, EMG_epoch),
            train_subjects_ids = train_subjects_ids,
            config = config,
            mean_epochs = avg_fold_epochs,
            device = device,
            print_config = print_config,
            single_or_fusion = 'fusion'
        )
        
        # Prepara test dataset for inference
        test_loader  = DataLoader(test_dataset_ins, batch_size = config["batch_size"], shuffle=False)
        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = retrain_model, test_loader = test_loader, criterion = retrain_criterion, device = device)

        study.add_observation(
            trial = trial,
            objective = avg_fold_vloss,
            iteration = 0
        )

        study.finalize(trial)

        if avg_fold_vloss < global_best_vloss :
            global_best_vloss = avg_fold_vloss
            print(f'Global best avg vloss: {avg_fold_vloss}. Test accuracy: {test_acc}. avg val accuracy: {np.mean(fold_val_accs)}')
            
            torch.save({
                "model_name": model_name,
                "sensor_name": 'fusion',
                "model_state": retrain_model.state_dict(),
                "model_args": model_config, 
                "optimizer_state_dict": retrain_optimizer.state_dict(),
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))


        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = fold_epochs,
            train_loss = fold_train_losses,
            val_loss = fold_val_losses,
            val_acc = fold_val_accs,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def singleNet_classfication_independent(subject_name : str | list, sherpa_log_folder : str = 'subject_dependent/SingleNet_LSTM_EMG', sensor_name : str = None, model_name : str = 'SingleNet_LSTM'):
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    check_model(model_name = model_name, sensor_name = sensor_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'all_subjects'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    sensor_name = str.upper(sensor_name)

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    train_eval_ins = SingleNet_train_eval()
    model_handler_ins = SingleNetHandler(model_name = model_name, sensor_name = sensor_name)

    if sensor_name == 'EMG':
        EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    elif sensor_name == 'EEG':
        EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    else:
        raise ValueError('data_type must be either EMG or EEG')

    #====================#
    # Load Training data #
    #====================#
    VALIDATE_SUBJECTS = ['subject_3', 'subject_5', 'subject_7']
    TEST_SUBJECT = ['subject_8']
    data = {
    'train': {'X_index': [], 'X_thumb': []},
    'val':   {'X_index': [], 'X_thumb': []},
    'test':  {'X_index': [], 'X_thumb': []}
    }

    for subj in subject_name:
        if sensor_name == 'EMG':
            X_i, _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = 'index', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
            X_t, _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = 'thumb', EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)
        else:
            X_i, _ = load_ins.load_EEG_data(subject_name = subj, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)
            X_t, _ = load_ins.load_EEG_data(subject_name = subj, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EEG_ins.preprocessing_routine, EEG_useable_channels = EEG_USEABLE_CHANNELS)

        if subj in TEST_SUBJECT:
            split = 'test'
        elif subj in VALIDATE_SUBJECTS:
            split = 'val'
        else:
            split = 'train'

        data[split]['X_index'].append(X_i)
        data[split]['X_thumb'].append(X_t)
    
    # Across multiple subjects
    for split in data:
        for key in data[split]:
            data[split][key] = np.concatenate(data[split][key], axis=0)
    
    FREQ = RMS_FREQ if sensor_name == 'EMG' else EEG_FREQ

    X_train, y_train = split_ins._build_split(epoch_index = data['train']['X_index'],
                                              epoch_thumb = data['train']['X_thumb'],
                                              index_trials_indices = slice(None),
                                              thumb_trials_indices = slice(None),
                                              fs = FREQ)

    X_val, y_val = split_ins._build_split(epoch_index = data['val']['X_index'],
                                          epoch_thumb = data['val']['X_thumb'],
                                          index_trials_indices = slice(None),
                                          thumb_trials_indices = slice(None),
                                          fs = FREQ)
    
    X_test, y_test = split_ins._build_split(epoch_index = data['test']['X_index'],
                                            epoch_thumb = data['test']['X_index'],
                                            index_trials_indices = slice(None),
                                            thumb_trials_indices = slice(None),
                                            fs = FREQ)
        
    _, _, num_channels = X_train.shape
    
    #=================#
    # Single datasets #
    #=================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = SingleManageDataset(X_train, y_train, data_type = sensor_name)
    print('Validation dataset shapes:')
    val_dataset_ins = SingleManageDataset(X_val, y_val, data_type = sensor_name)
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = sensor_name)
    
    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 75
    DATA_CH = num_channels
    NUM_CLASSES = 5 if sensor_name == 'EMG' else 3
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial = trial,
            input_dim = DATA_CH,
            TOTAL_CLASSES = NUM_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )
        print(model_config)
        print(train_config)

        #=================#
        # Single datasets #
        #=================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)

        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validate_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)
        
        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss
            
            torch.save({
                "model_name": model_name,
                "sensor_name": sensor_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

def fusionNet_classfication_independent(subject_name : list, sherpa_log_folder : str = 'SingleNet_LSTM_EMG', model_name : str = 'FusionNet_LSTM', num_motions = 2):
    '''
    Train a model with EEG and EMG across subjects. 
    Subjects are clearly separated between traning, validation and test split
    
    Parameters
    -----------
    subject_name : list
        List of all subjects to be included
    sherpa_log_folder : str
        Path to where the model and loggings need to be saved in the 'src/experiment/data' directory
    model_name : str
        Select between models.\n
        Options:
        1) FusionNet_LSTM
        2) FusionNet_CNN_LSTM
        3) FusionNet_CNN_LSTM_ATTENSION
    '''
    # When chancing between EEG and EMG
    # preprocessing instance
    # Load function
    check_model(model_name = model_name, sensor_name = None, num_motions = num_motions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available
    print(f"Using device: {device}")
    print("Pin memory set to:", pin_memory)

    LOG_NAME = 'all_subjects'
    log_dir = Path(__file__).resolve().parent / f'loggings/{sherpa_log_folder}/{LOG_NAME}'         # Path(__file__).resolve() -> Absolute path to this file
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'

    #==========================#
    # NOTE: Tensorboard config #
    #==========================#
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                                        # Use when having tensorboard
    os.makedirs(log_dir, exist_ok=False)                                                          # use without tensorboard

    logger_ins = ExperimentLogger(save_path = log_dir)
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    train_eval_ins = FusionNet_train_eval()
    model_handler_ins = FusionNetHandler(model_name = model_name)

    #====================#
    # Load Training data #
    #====================#
    VALIDATE_SUBJECTS = ['subject_1']#['subject_3', 'subject_5', 'subject_7']
    TEST_SUBJECT = ['subject_2']#['subject_8']
    data = {
    'train': {'EEG_index': [], 'EEG_thumb': [], 'EMG_index': [], 'EMG_thumb': []},
    'val':   {'EEG_index': [], 'EEG_thumb': [], 'EMG_index': [], 'EMG_thumb': []},
    'test':  {'EEG_index': [], 'EEG_thumb': [], 'EMG_index': [], 'EMG_thumb': []}
    }

    for subj in subject_name:

        EEG_i, EMG_i, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj, finger_name = 'index', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)
        EEG_t, EMG_t, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj, finger_name = 'thumb', reject_config_dict = REJECT_CONFIG_DICT, EEG_preprocessing_func = EEG_ins.preprocessing_routine, EMG_preprocessing_func = EMG_ins.preprocessing_routine, EMG_config_dict = EMG_CONFIG_DICT, EEG_useable_channels = EEG_USEABLE_CHANNELS)

        if subj in TEST_SUBJECT:
            split = 'test'
        elif subj in VALIDATE_SUBJECTS:
            split = 'val'
        else:
            split = 'train'

        data[split]['EEG_index'].append(EEG_i)
        data[split]['EEG_thumb'].append(EEG_t)
        data[split]['EMG_index'].append(EMG_i)
        data[split]['EMG_thumb'].append(EMG_t)
    
    # Across multiple subjects
    for split in data:
        for key in data[split]:
            data[split][key] = np.concatenate(data[split][key], axis=0)
    
    def _build_split_recall(train_index, train_thumb, val_index, val_thumb, test_index, test_thumb, fs):

        X_train, y_train = split_ins._build_split(epoch_index = train_index,
                                                epoch_thumb = train_thumb,
                                                index_trials_indices = slice(None),
                                                thumb_trials_indices = slice(None),
                                                fs = fs)

        X_val, y_val = split_ins._build_split(epoch_index = val_index,
                                            epoch_thumb = val_thumb,
                                            index_trials_indices = slice(None),
                                            thumb_trials_indices = slice(None),
                                            fs = fs)
        
        X_test, y_test = split_ins._build_split(epoch_index = test_index,
                                                epoch_thumb = test_thumb,
                                                index_trials_indices = slice(None),
                                                thumb_trials_indices = slice(None),
                                                fs = fs)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    X_EEG_train, X_EEG_val, X_EEG_test, y_EEG_train, y_EEG_val, y_EEG_test = _build_split_recall(
        train_index = data['train']['EEG_index'],
        train_thumb = data['train']['EEG_thumb'],
        val_index = data['val']['EEG_index'],
        val_thumb = data['val']['EEG_thumb'],
        test_index = data['test']['EEG_index'],
        test_thumb = data['test']['EEG_thumb'],
        fs = EEG_FREQ)
    
    X_EMG_train, X_EMG_val, X_EMG_test, y_EMG_train, y_EMG_val, y_EMG_test = _build_split_recall(
        train_index = data['train']['EMG_index'],
        train_thumb = data['train']['EMG_thumb'],
        val_index = data['val']['EMG_index'],
        val_thumb = data['val']['EMG_thumb'],
        test_index = data['test']['EMG_index'],
        test_thumb = data['test']['EMG_thumb'],
        fs = RMS_FREQ)
    
    _, _, EEG_num_channels = X_EEG_train.shape
    _, _, EMG_num_channels = X_EMG_train.shape

    #=======================#
    # Multi fusion datasets #
    #=======================#
    print('\nTraining dataset shapes:')
    train_dataset_ins = MultiManageDataset(X_EEG_train, X_EMG_train, y_EEG_train, y_EMG_train)
    print('Validation dataset shapes:')
    val_dataset_ins = MultiManageDataset(X_EEG_val, X_EMG_val, y_EEG_val, y_EMG_val)
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    #========================================================#
    # THESE PARAMETERS ARE CHANCEABLE, DEPENDING ON THE TASK #
    #========================================================#
    MAX_NUM_TRIALS = 100             # 75 - 250 (simply to max) 
    NUM_INITIAL_DATA_POINTS = 75
    EEG_CH = EEG_num_channels
    EMG_CH = EMG_num_channels
    EEG_CLASSES = 3
    EMG_CLASSES = 5
    TOTAL_CLASSES = EMG_CLASSES
    NUM_EPOCHS = 250                 # 150 - 200
    PATIENCE = 25                   # Early stopping patience - 25
    
    #===========#
    # Constants #
    #===========#
    global_best_vloss = float("inf")                # Used to only save one model.

    #====================================#
    # SHERPA Hyperparameter Optimazation #
    #====================================#

    parameters = model_handler_ins.get_hyperparameters()
    
    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials = MAX_NUM_TRIALS)
    algorithm = sherpa.algorithms.GPyOpt(
        max_num_trials = MAX_NUM_TRIALS,
        acquisition_type = 'EI',                     # Expected improvement
        num_initial_data_points = NUM_INITIAL_DATA_POINTS                 # Number of hyperparameter configurations before model learns
    )
    # Study represents the hyperparameter optimization itself
    study = sherpa.Study(
        parameters = parameters,
        algorithm = algorithm,
        lower_is_better = True,
        disable_dashboard = True
    )

    for trial in study:
        model_config = model_handler_ins.build_model_config(
            trial,
            EEG_CH, EMG_CH,
            EEG_CLASSES, EMG_CLASSES,
            TOTAL_CLASSES
        )
        train_config = model_handler_ins.build_training_config(
            trial = trial
        )
        print(model_config)
        print(train_config)
        #=======================#
        # Multi fusion datasets #
        #=======================#
        model = model_handler_ins.get_model(config = model_config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = train_config['lr'], weight_decay = train_config['weight_decay'])

        # DataLoaders (update batch_size)
        train_loader = DataLoader(train_dataset_ins, batch_size = train_config['batch_size'], shuffle = True, pin_memory = pin_memory, num_workers = 0)
        val_loader = DataLoader(val_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        test_loader = DataLoader(test_dataset_ins, batch_size = train_config['batch_size'], shuffle = False, pin_memory = pin_memory, num_workers = 0)
        
        best_train_loss = None
        best_val_loss = float("inf")
        best_val_acc = None

        best_epoch = 0
        best_state_dict = None
        early_stopping_counter = 0

        #===================================#
        # NOTE: Tensorboard config          #
        #   Enable all if using tensorboard #
        #===================================#
        # log_folder = os.path.join(log_dir, f"trial_{trial.id}")               
        # os.makedirs(log_folder, exist_ok=False)
        # writer = SummaryWriter(os.path.join(log_folder, 'trial_{}_timestamp_{}'.format(trial.id, timestamp)))

        for epoch in range(NUM_EPOCHS):

            # Train model
            avg_train_loss = train_eval_ins.train_one_epoch(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)

            # Validate model
            avg_vloss, vacc, _ = train_eval_ins.validate_one_epoch(model = model, val_loader = val_loader, criterion = criterion, device = device)

            # Tensor Board logging
            # writer.add_scalars('Loss', { 'Training' : avg_train_loss, 'Validation' : avg_vloss }, epoch + 1)
            # writer.add_scalars('Accuracy Validation', {'Validation' : vacc }, epoch + 1)
            # writer.flush()

            study.add_observation(trial = trial,
                                iteration = epoch,
                                objective = avg_vloss)

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_epoch = epoch

                best_state_dict = deepcopy(model.state_dict())
                best_optimizer_dict = deepcopy(optimizer.state_dict())
 
                best_train_loss = avg_train_loss
                best_val_acc = vacc

                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

                if early_stopping_counter >= PATIENCE:
                    break

            print(
                f'Trial {trial.id}/{MAX_NUM_TRIALS} | '
                f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train {avg_train_loss:.4f} | '
                f'Val {avg_vloss:.4f} | '
                f'Acc {vacc:.2f} |',
                f'Early stopping {early_stopping_counter} |',
                end='\r',
                flush=True
            )

        model.load_state_dict(best_state_dict)

        avg_test_loss, test_acc, predictions, labels = train_eval_ins.inference_one_epoch(model = model, test_loader = test_loader, criterion = criterion, device = device)
        
        logger_ins.log_trial(
            trial_id=trial.id,
            hyperparams = trial.parameters,
            best_epoch = best_epoch,
            train_loss = best_train_loss,
            val_loss = best_val_loss,
            val_acc = best_val_acc,
            test_loss = avg_test_loss,
            test_acc = test_acc,
            preds = predictions,
            labels = labels)

        if best_val_loss < global_best_vloss :
            global_best_vloss = best_val_loss

            torch.save({
                "model_name": model_name,
                "model_state": best_state_dict,
                "model_args": model_config, 
                "optimizer_state_dict": best_optimizer_dict,
                "hyperparameters": trial.parameters,}, 
                r'{}\model.pth'.format(log_dir))
            
        # writer.close()                            # NOTE: Enable with tensorboard
        study.finalize(trial, status = 'COMPLETED')

#================#
# Analyse models #
#================#
def inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG', include_all = False):

    #=======================#
    # Include all in a list #
    #=======================#
    high_vloss = []
    high_acc = []
    subjs_nr = []
    high_vloss_loss = []
    high_vloss_acc = []

    if include_all:
        cms = []
        for subj_nr in range(17):
            sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/subject_{subj_nr}/SHERPA_results.pt"

            if not os.path.exists(sherpa_info_path):
                continue

            data = torch.load(sherpa_info_path, weights_only=False)
            
            best_vloss = min(
                data["trials"],
                key=lambda x: x["validation_loss"]
            )
            best_tacc = max(
                data['trials'],
                key = lambda x: x['test_accuracy']
            )

            cm = confusion_matrix(best_tacc['labels'], best_tacc['predictions'])
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row normalize
            cms.append(cm)

            high_vloss.append(best_vloss["test_accuracy"])
            high_vloss_acc.append(best_vloss["validation_accuracy"])
            high_vloss_loss.append(best_vloss["validation_loss"])

            high_acc.append(best_tacc["test_accuracy"])
            subjs_nr.append(subj_nr)

        print(f'Subject:        {subjs_nr}')
        print('TAcc - lowest Vloss',", ".join(f"{float(x):.2f}" for x in high_vloss))
        print("VAcc - lowest Vloss",", ".join(f"{float(x):.2f}" for x in high_vloss_acc))
        print("Vloss - lowest Vloss",", ".join(f"{float(x):.2f}" for x in high_vloss_loss))
        print("Acc - highest Ttest",", ".join(f"{float(x):.2f}" for x in high_acc))
        print('Confusion matrices for lowest validation loss:\n ', np.mean(cms, axis=0))
        print()
        return 0
    
    sherpa_info_path = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}/SHERPA_results.pt"
    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    data = torch.load(sherpa_info_path, weights_only=False)

    # Extract test accuracies
    # Extract test accuracies
    '''acc_list = [trial['validation_loss'] for trial in data['trials']]

    # Get indices sorted from highest → lowest accuracy
    sorted_indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=False)

    # Iterate over trials in sorted order
    for rank, idx in enumerate(sorted_indices):
        trial = data['trials'][idx]

        print('Rank:', rank + 1)
        print('Trial:', idx + 1)
        print('Epochs', trial['best_epoch'])
        print('Training loss:' , trial['training_loss'])
        print('Validation loss:', trial['validation_loss'])
        print('Validation accuracy:', trial['validation_accuracy'])
        print('Test accuracy:', trial['test_accuracy'])
        print('Hyperparameters:\n', trial['hyperparameters'], '\n')
        if rank > 10:
            break'''
    
    # print('Last ten')
    data_len = len(data['trials'])
    print('Len of trials:', data_len)
    for idx in range(data_len - 10, data_len):
        trial = data['trials'][idx]

        # print('Rank:', rank + 1)
        print('Trial:', idx + 1)
        print('Epochs', trial['best_epoch'])
        print('Training loss:' , trial['training_loss'])
        print('Validation loss:', trial['validation_loss'])
        print('Validation mean: ', np.mean(trial['validation_loss']))
        print('Validation accuracy:', trial['validation_accuracy'])
        print('Test accuracy:', trial['test_accuracy'])
        print('Hyperparameters:\n', trial['hyperparameters'], '\n')

    best_vloss = min(
        data["trials"],
        key=lambda x: np.mean(x["validation_loss"])
    )
    best_tacc = max(
        data['trials'],
        key = lambda x: x['test_accuracy']
    )

    print(f'\n---------{subject_name}-----------')
    for best_name, best_value in zip(['lowest validation loss', 'highest test accuracy'], [best_vloss, best_tacc]):
        cm = confusion_matrix(best_value['labels'], best_value['predictions']) 
        print(f'For {best_name}')
        print('         Best trial ID: ', best_value['trial_id'])
        print('         Stoped at epoch', best_value['best_epoch'])
        print('         Training loss:' , best_value['training_loss'])
        print('         validation loss', best_value['validation_loss'])
        print('         validation mean:', np.mean(best_value['validation_loss']))
        print('         validation acc' , best_value['validation_accuracy'])
        print('         Test accuracy: ', best_value["test_accuracy"])
        print('         Hyperparameter: ', best_value['hyperparameters'])
        np.set_printoptions(linewidth=200)
        print(cm)
        #print('\n')

def singleNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG', sensor_name = 'None', num_motions = 0):
    # model_path_folder = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}"
    # sherpa_info_path = model_path_folder / 'SHERPA_results.pt'

    model_path_folder = Path(__file__).resolve().parent / f"loggings/subject_independent/{sherpa_log_folder}"
    sherpa_info_path = model_path_folder / f'{subject_name}/SHERPA_results.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available

    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    
    data = torch.load(sherpa_info_path, weights_only=False)
        
    best = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )

    best_trial_id = best['trial_id']
    batch_size = best['hyperparameters']['batch_size']
    print(f'\n---------{subject_name}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('Hyperparameters :\n', best['hyperparameters'])
    print('\n')
    
    model_path = model_path_folder / f'{subject_name}/model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    print(checkpoint["model_args"])
    model_args = checkpoint["model_args"]

    # model_interference = SingleNet_LSTM(**checkpoint["model_args"])
    # model_interference = SingleNet_LSTM(**model_args)
    model_interference = SingleNet_CNN_LSTM(**model_args)

    model_interference.load_state_dict(checkpoint["model_state"])
    model_interference.to(device)

    #===================#
    # Load Test dataset #
    #===================#
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    # EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    if sensor_name == 'EMG':
        pass
    else:
        raise ValueError('Can currently only be EMG modaility')

    if num_motions == 2:
        motion_list = ['index', 'thumb']
    elif num_motions == 3:
        motion_list = ['index', 'thumb', 'pinchGrip']
    elif num_motions == 7:
        motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip']
    else:
        raise ValueError(f'num_motions is not valid : {num_motions}')

    TEST_SUBJECT = ['subject_8']    
    EEG_epoch = {}
    EMG_epoch = {}

    motion_list = ['index', 'thumb']
    for subj in TEST_SUBJECT:
        EEG_epoch[subj] = {}
        EMG_epoch[subj] = {}

        for ml in motion_list:
            print(f'Load for {subj} for {ml}')
            emg_temp, _, _ = load_ins.load_EMG_data(subject_name = subj, finger_name = ml, EMG_config_dict = EMG_CONFIG_DICT, reject_config_dict = REJECT_CONFIG_DICT, preprocessing_func = EMG_ins.preprocessing_routine)

            EMG_epoch[subj][ml] = emg_temp

    # Prepare test dataset
    X_EMG_test, y_EMG_test = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = TEST_SUBJECT, fs = RMS_FREQ)
    
    
    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_EMG_test, y_EMG_test, data_type = 'EMG')

    test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

    #===================#
    # Perform inference #
    #===================#
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    # all_context = []

    model_interference.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _, _ = model_interference(inputs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            # all_context.append(context.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()
        # all_context = torch.cat(all_context).numpy()

        #==========#
        # Analysis #
        #==========#

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        # print(cm_norm)

        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")

    _plot_tsne_context(context = all_logits, labels = y_EMG_test, perplexity = 40, n_iter = 1000, random_state = 42, num_motions=num_motions)
    # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    score = silhouette_score(all_logits, y_EMG_test)
    print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping

def real_time_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG', sensor_name = '', num_motions = 7):
    # model_path_folder = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}"
    # sherpa_info_path = model_path_folder / 'SHERPA_results.pt'

    model_path_folder = Path(__file__).resolve().parent / f"loggings/real_time/{sherpa_log_folder}"
    sherpa_info_path = model_path_folder / f'{subject_name}/SHERPA_results.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available

    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    
    data = torch.load(sherpa_info_path, weights_only=False)
        
    best = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )

    best_trial_id = best['trial_id']
    batch_size = best['hyperparameters']['batch_size']
    print(f'\n---------{subject_name}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('Hyperparameters :\n', best['hyperparameters'])
    print('\n')
    
    model_path = model_path_folder / f'{subject_name}/model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    print(checkpoint["model_args"])
    model_args = checkpoint["model_args"]

    model_interference = SingleNet_CNN_LSTM(**model_args)

    model_interference.load_state_dict(checkpoint["model_state"])
    model_interference.to(device)

    #===================#
    # Load Test dataset #
    #===================#
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMGStreamProcessor(fs = EMG_FREQ, lowcut=EMG_LOWCUT, highcut=EMG_HIGHCUT,
                                 reject_config_dict = REJECT_CONFIG_DICT,
                                rms_window = 500, rms_step = 50,
                                hampel_window = 100, hampel_sigma = 2, base_dir = data_dir)
    
    if sensor_name == 'EMG':
        pass
    else:
        raise ValueError('Can currently only be EMG modaility')
    
    X_epoch = {}
    X_labels = {}
    if num_motions == 7:
        motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip']
    elif num_motions == 3:
        motion_list = ['index', 'thumb', 'pinchGrip']
    elif num_motions == 2:
        motion_list = ['index']
    
    SUBJECTS_IDs = ['subject_0']

    for subj in SUBJECTS_IDs:
        X_epoch[subj] = {}
        X_labels[subj] = {}

        for ml in motion_list:
            data, num_epochs = EMG_ins.load_subject_data(subj = subj, finger = ml, modality = sensor_name, trim_period = TRIM_PERIOD, trial_period = TRIAL_PERIOD)    

            # Trial-level split
            train_idx, val_idx, test_idx = split_ins._split_trials(num_trials = num_epochs, train_ratio = 0.7)

            # Split each window into train, val, test sets
            X_epoch[subj][ml] = {}
            X_labels[subj][ml] = {}

            for split_name, split_indicies in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
                split_data = data[split_indicies]
                
                X_e, y_e = EMG_ins.relabel_windows(
                    epochs = split_data,
                    window_samples = 1000,
                    step_samples = 200,
                    fs = EMG_FREQ,
                    labels = ['rest', 'contract', 'release']
                )

                X_epoch[subj][ml][split_name] = X_e
                X_labels[subj][ml][split_name] = y_e
    
    X_filt, mu, sigma = normalize_global_per_channel(X_epoch, train_subject_ids = SUBJECTS_IDs)
    
    # Normalize
    mu_saved = np.load(model_path_folder / f"{subject_name}/mu.npy")
    sigma_saved = np.load(model_path_folder / f"{subject_name}/sigma.npy")
    
    print(f'mu = {mu}, sigma = {sigma}')
    print(f'Saved mu = {mu_saved}, saved sigma = {sigma_saved}')

    datasets = {}
    for split in ['train', 'val', 'test']:
        X, y = split_ins.build_dataset_window_relabel(
            X_epoch = X_filt,
            X_labels = X_labels,
            subjects = SUBJECTS_IDs,
            split = split
        )      

        datasets[split] = (X, y) 

    X_test, y_test = datasets['test'] 

    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = SingleManageDataset(X_test, y_test, data_type = sensor_name)

    test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = True, pin_memory = pin_memory, num_workers = 0)

    #===================#
    # Perform inference #
    #===================#
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    # all_context = []

    # -------------------------
    # Evaluation
    # -------------------------
    model_interference.eval()

    incorrect_samples = []
    correct_samples = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _, _ = model_interference(inputs)

            preds = torch.argmax(logits, dim=1)

            # Accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Store all outputs
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

            # -------------------------
            # Save incorrect predictions
            # -------------------------
            wrong_mask = preds != labels
            right_mask = preds == labels

            wrong_input = inputs[wrong_mask]
            wrong_preds = preds[wrong_mask]
            wrong_labels = labels[wrong_mask]
            
            right_input = inputs[right_mask]
            right_preds = preds[right_mask]
            right_labels = labels[right_mask]

            for x, p, y in zip(wrong_input, wrong_preds, wrong_labels):

                incorrect_samples.append({
                    "signal": x.cpu().numpy(),
                    "pred": int(p.cpu().item()),
                    "label": int(y.cpu().item())
                })

            for x, p, y in zip(right_input, right_preds, right_labels):
                correct_samples.append({
                    "signal": x.cpu().numpy(),
                    "pred": int(p.cpu().item()),
                    "label": int(y.cpu().item())
                })

    # Convert
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()

    print(f"Total incorrect samples: {len(incorrect_samples)}")

    # ==========================================================
    # Plot 2x4 incorrect samples
    # ==========================================================
    '''
    plots_per_figure = 8
    plots_to_inspect = 4 * plots_per_figure
    map_to_labels = ['index contract', 'index release', 'thumb contract', 'thumb release', 'pinch contract', 'pinch release', 'rest']

    for start_idx in range(0, plots_to_inspect, plots_per_figure):

        for samples in [incorrect_samples, correct_samples]:
            fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
            axes = axes.flatten()

            batch = samples[start_idx:start_idx + plots_per_figure]

            for i, sample in enumerate(batch):

                signal = sample["signal"]
                pred = sample["pred"]
                label = sample["label"]

                ax = axes[i]

                # ---------------------------------------
                # Handle signal shape
                # ---------------------------------------
                if signal.shape[0] < signal.shape[1]:
                    # (channels, time)
                    for ch in range(signal.shape[0]):
                        ax.plot(signal[ch], linewidth=1)

                else:
                    # (time, channels)
                    for ch in range(signal.shape[1]):
                        ax.plot(signal[:, ch], linewidth=1)

                ax.set_title(f"True: {map_to_labels[label]} | Pred: {map_to_labels[pred]}")
                ax.grid(True)

            # Hide unused axes
            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
        plt.show()
    
    model_interference.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _, _ = model_interference(inputs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            # all_context.append(context.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()
        # all_context = torch.cat(all_context).numpy()'''

        #==========#
        # Analysis #
        #==========#

        # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    np.set_printoptions(linewidth=200)
    print(cm)

    # cm_norm = cm / cm.sum(axis=1, keepdims=True)
    # print(cm_norm)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
        
    # _plot_tsne_context(context = all_logits, labels = y, perplexity = 40, n_iter = 1000, random_state = 42)
    # # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    # score = silhouette_score(all_logits, y)
    # print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping


def fusionNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'SingleNet_LSTM_EMG'):
    model_path_folder = Path(__file__).resolve().parent / f"loggings/{sherpa_log_folder}/{subject_name}"
    sherpa_info_path = model_path_folder / 'SHERPA_results.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()              # Use pin_memory if CUDA is available

    if not os.path.exists(sherpa_info_path):
        raise FileExistsError(sherpa_info_path)
    
    data = torch.load(sherpa_info_path, weights_only=False)
        
    best = min(
        data["trials"],
        key=lambda x: x["validation_loss"]
    )

    best_trial_id = best['trial_id']
    batch_size = best['hyperparameters']['batch_size']
    print(f'\n---------{subject_name}-----------')
    print('Best trial ID: ', best_trial_id)
    print('Stoped at epoch', best['best_epoch'])
    print('Training loss:' , best['training_loss'])
    print('validation loss', best['validation_loss'])
    print('Test accuracy: ', best["test_accuracy"])
    print('\n')
    
    model_path = model_path_folder / 'model.pth'
    checkpoint = torch.load(f = model_path, map_location = device)

    model_args = checkpoint["model_args"]

    # Add missing arguments
    # model_args["eeg_output_dim"] = 3
    # model_args["emg_output_dim"] = 5
    # model_args["dense_fusion_layer"] = 16

    model_interference = FusionNet_LSTM(**model_args)             # NOTE : **checkpoint["model_args"]
    
    model_interference.load_state_dict(checkpoint["model_state"])
    model_interference.to(device)

    #===================#
    # Load Test dataset #
    #===================#
    data_dir = Path(__file__).resolve().parents[2] / 'src/experiment/data'
    
    load_ins = load_datasets(base_dir = data_dir)
    split_ins = Manage3Split(seed = SEED)
    EMG_ins = EMG_preprocessing(fs = EMG_FREQ, bandpass_lowcut = EMG_LOWCUT, bandpass_highcut = EMG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)
    EEG_ins = EEG_preprocessing(fs = EEG_FREQ, bandpass_lowcut = EEG_LOWCUT, bandpass_highcut = EEG_HIGHCUT, trial_period = TRIAL_PERIOD, trim_period = TRIM_PERIOD)

    TEST_SUBJECT = ['subject_8']    
    EEG_epoch = {}
    EMG_epoch = {}

    motion_list = ['index', 'thumb']
    for subj in TEST_SUBJECT:
        EEG_epoch[subj] = {}
        EMG_epoch[subj] = {}

        for ml in motion_list:
            print(f'Load for {subj} for {ml}')
            eeg_temp, emg_temp, _, _ = load_ins.load_EEG_EMG_data(subject_name = subj,
                                                                    finger_name = ml,
                                                                    reject_config_dict = REJECT_CONFIG_DICT,
                                                                    EEG_preprocessing_func = EEG_ins.preprocessing_routine,
                                                                    EMG_preprocessing_func = EMG_ins.preprocessing_routine,
                                                                    EMG_config_dict = EMG_CONFIG_DICT,
                                                                    EEG_useable_channels = EEG_USEABLE_CHANNELS)
            EEG_epoch[subj][ml] = eeg_temp
            EMG_epoch[subj][ml] = emg_temp

    # Prepare test dataset
    X_EEG_test, y_EEG_test = split_ins.build_dataset_from_subjects(X_epoch = EEG_epoch, subjects = TEST_SUBJECT, fs = EEG_FREQ)
    X_EMG_test, y_EMG_test = split_ins.build_dataset_from_subjects(X_epoch = EMG_epoch, subjects = TEST_SUBJECT, fs = RMS_FREQ)
    
    #=================#
    # Single datasets #
    #=================#
    print('Testing dataset shapes:')
    test_dataset_ins = MultiManageDataset(X_EEG_test, X_EMG_test, y_EEG_test, y_EMG_test)

    test_loader = DataLoader(test_dataset_ins, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = 0)

    #===================#
    # Perform inference #
    #===================#

    correct_fusion = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    # all_context = []
    confidence_eeg_all = []
    confidence_emg_all = []
    confidence_final_all = []

    model_interference.eval()

    criterion = nn.CrossEntropyLoss()
    loss_eeg_all = 0
    loss_emg_all = 0
    loss_final_all = 0
    loss_all = 0
    correct_eeg = 0
    correct_emg = 0
    

    with torch.no_grad():
        for eeg, emg, eeg_lab, emg_lab in test_loader:
            X_eeg, X_emg, y_eeg, y_emg = eeg.to(device), emg.to(device), eeg_lab.to(device), emg_lab.to(device)

            # Forward pass
            final_logits, eeg_logits, emg_logits = model_interference(eeg = X_eeg, emg = X_emg)
            
            # Predicted class index
            _, fusion_pred = torch.max(final_logits, dim=1)             # index of max value (predicted class)

            # Compute the loss and its gradients
            loss_final = criterion(final_logits, y_emg)        # EMG has all 5 lables (contract per finger, release per finger, rest all fingers)
            loss_eeg   = criterion(eeg_logits, y_eeg)          # EEG has only 3 lables (contract, release, rest)
            loss_emg   = criterion(emg_logits, y_emg)  

            loss = loss_final + 0.3 * loss_eeg + 0.3 * loss_emg     # Only used for training optimization

            # Accuracy statistics
            total += y_emg.size(0)
            correct_fusion += (fusion_pred == y_emg).sum().item()
            
            # Store outputs for confusion matrix etc.
            all_preds.append(fusion_pred.cpu())
            all_labels.append(y_emg.cpu())
            all_logits.append(final_logits.cpu())
            # all_context.append(context.cpu())

            #=====================================#
            # Inspect contribution of EEG and EMG #
            #=====================================#
            eeg_pred = torch.argmax(eeg_logits, dim=1)
            emg_pred = torch.argmax(emg_logits, dim=1)

            # Calculate confidence
            eeg_conf, _ = torch.max(torch.softmax(eeg_logits, dim=1), dim=1)
            emg_conf, _ = torch.max(torch.softmax(emg_logits, dim=1), dim=1)
            final_conf, _ = torch.max(torch.softmax(final_logits, dim=1), dim=1)
            confidence_eeg_all.append(eeg_conf)
            confidence_emg_all.append(emg_conf)
            confidence_final_all.append(final_conf)

            correct_eeg += (eeg_pred == y_eeg).sum().item()
            correct_emg += (emg_pred == y_emg).sum().item()
            loss_final_all += loss_final.item()
            loss_eeg_all += loss_eeg.item()
            loss_emg_all += loss_emg.item()
            loss_all += loss.item()

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()

        confidence_eeg_all = torch.cat(confidence_eeg_all).numpy()
        confidence_emg_all = torch.cat(confidence_emg_all).numpy()
        confidence_final_all = torch.cat(confidence_final_all).numpy()
        # all_context = torch.cat(all_context).numpy()

        num_batches = len(test_loader)
        print('\n------LOSSES-------\n')
        print(f"Avg EEG loss   : {loss_eeg_all / num_batches :.4f}")
        print(f"Avg EMG loss   : {loss_emg_all / num_batches :.4f}")
        print(f"Avg final loss : {loss_final_all / num_batches :.4f}")
        print(f"Avg comb loss  : {loss_all / num_batches :.4f}")
        print('\n-------Accuracies---------\n')
        print(f'EEG accuracy : {(correct_eeg / total) * 100 :.2f}')
        print(f'EMG accuracy : {(correct_emg / total) * 100 :.2f}')
        print(f"Fusion accuracy: {(correct_fusion / total) * 100 :.2f}")
        print('\n-------Confidence---------\n')
        print(f'EEG confidence : {confidence_eeg_all.mean() * 100 :.2f}')
        print(f'EMG confidence : {confidence_emg_all.mean() * 100 :.2f}')
        print(f'Fusion confidence: {confidence_final_all.mean() * 100 :.2f}')
        #==========#
        # Analysis #
        #==========#

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        print(cm_norm)
    
    _plot_tsne_context(context = all_logits, labels = y_EMG_test, perplexity = 40, n_iter = 1000, random_state = 42)
    # plot_tsne_context(context = all_context, labels = y_test, perplexity = 40, n_iter = 1000, random_state = 42)

    score = silhouette_score(all_logits, y_EMG_test)
    print(score)

    # ~0.5 → good separation
    # ~0.2 → weak separation
    # ~0 → no separation
    # <0 → overlapping

def _plot_tsne_context(
    context,
    labels,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    title="t-SNE of Attention Context",
    figsize=(8, 6),
    num_motions = 0
):
    """
    Plot t-SNE embedding of attention context vectors.

    Args:
        context (torch.Tensor or np.ndarray):
            Shape (n_samples, feature_dim)
        labels (torch.Tensor or np.ndarray):
            Shape (n_samples,)
        perplexity (int):
            t-SNE perplexity (typ. 5-50)
        n_iter (int):
            Number of optimization iterations
        random_state (int):
            Seed for reproducibility
        title (str):
            Plot title
        figsize (tuple):
            Figure size
    """

    # ---- detach safely ----
    if hasattr(context, "detach"):
        X = context.detach().cpu().numpy()
    else:
        X = np.asarray(context)

    if hasattr(labels, "detach"):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_jobs=1
    )

    X_embedded = tsne.fit_transform(X)

    import matplotlib.colors as mcolors

    # ---- convert labels to integer classes ----
    # Example: map unique labels to 0..4
    
    unique_classes = np.unique(y)
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_int = np.vectorize(class_mapping.get)(y)

    num_classes = len(unique_classes)

    # ---- discrete colormap ----
    cmap = plt.get_cmap("tab10", num_classes)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, num_classes + 0.5, 1),
        ncolors=num_classes
    )

    # ---- plot ----
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_int,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=25
    )
    # class_names = ['Index contract', 'Index release', 'Thumb contract', 'Thumb release', 'Rest']
    actions = ['contract', 'release']
    if num_motions == 7:
        limbs = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinch', 'cylinder']
    elif num_motions == 3:
        limbs = ['index', 'thumb', 'pinch']
    elif num_motions == 2:
        limbs = ['index', 'thumb']
    else:
        raise ValueError(f'unique classes outside of range : {num_motions}')
    class_names = []
    for limb in limbs:
        for action in actions:
            motion = limb + ' ' + action
            class_names.append(motion)
    
    class_names.append('rest')

    cbar = plt.colorbar(scatter, ticks=range(num_classes), label="Class")
    cbar.ax.set_yticklabels(class_names)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('SNE_subj_independent_LSTM+CNN.pdf', dpi = 400)
    plt.savefig('SNE_subj_independent_LSTM+CNN.png', dpi = 400)
    plt.show()

def plot_model_groups_subjects(
    subject_ids,
    eeg_acc,
    emg_acc,
    fusion_acc,
    title="Test Accuracy per Model and Subject",
    ylabel="Accuracy (%)"
):
    """
    Plot grouped bars where each group is a model (EEG, EMG, Fusion)
    and each bar inside the group represents a subject.

    Parameters
    ----------
    subject_ids : list
        Example: ['S1','S2','S3','S4']

    eeg_acc : list
        Accuracy per subject for EEG model

    emg_acc : list
        Accuracy per subject for EMG model

    fusion_acc : list
        Accuracy per subject for Fusion model
    """

    models = ['LSTM-EEG', 'LSTM-EMG', 'LSTM-Fusion', 'CNN+LSTM-EEG', 'CNN+LSTM-EMG', 'CNN+LSTM-fusion', 'CNN+LSTM+Attension-EEG', 'CNN+LSTM+Attension-EMG', 'CNN+LSTM+Attension-fusion']
    data = [eeg_acc, emg_acc, fusion_acc]

    n_models = len(models)
    n_subjects = len(subject_ids)

    bar_width = 0.5 / n_subjects
    x = np.arange(n_models)

    plt.figure(figsize=(10,6))

    for i, subject in enumerate(subject_ids):
        subject_values = [data[m][i] for m in range(n_models)]
        offset = (i - (n_subjects - 1)/2) * bar_width

        plt.bar(
            x + offset,
            subject_values,
            bar_width,
            label=subject
        )

    plt.xticks(x, models)
    plt.ylabel(ylabel)
    plt.xlabel("Model Type")
    plt.title(title)
    plt.legend(title="Subjects")
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

def _plot_subject_accuracy_hierarchical(subject_ids, accuracies, architectures):
    """
    Plot grouped bars with hierarchical x-axis:
    Architecture -> Modality

    Parameters
    ----------
    subject_ids : list
        Example: ['S1','S2','S3']

    accuracies : dict
        Dictionary structured like:

        {
            "LSTM": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            },
            "CNN+LSTM": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            },
            "CNN+LSTM+Attention": {
                "EEG": [...],
                "EMG": [...],
                "Fusion": [...]
            }
        }

    architectures : list
        Example: ['LSTM','CNN+LSTM','CNN+LSTM+Attention']
    """

    #modalities = ['EEG']#["EEG", "EMG", "Fusion"]

    n_subjects = len(subject_ids)
    bar_width = 0.5 / n_subjects

    x_positions = []
    x_labels = []

    # Build positions
    pos = 0
    arch_centers = []

    for arch in architectures:

        start = pos

        # for mod in modalities:
        x_positions.append(pos)
        # x_labels.append(mod)
        pos += 1

        end = pos - 1
        arch_centers.append((start + end) / 2)

        pos += 0.1  # spacing between architectures

    plt.figure()

    # Plot bars per subject
    for i, subject in enumerate(subject_ids):

        offset = (i - (n_subjects - 1)/2) * bar_width

        values = []

        for arch in architectures:
            # for mod in modalities:
            values.append(accuracies[arch]['EEG'][i])

        plt.bar(
            np.array(x_positions) + offset,
            values,
            bar_width,
            label=f'Subject {i+1}'
        )

    plt.xticks(x_positions, x_labels)
    plt.ylabel("Accuracy (%)")
    plt.yticks(np.arange(0, 100.1, 10))
    plt.ylim([0, 100])
    # plt.xlabel("Model / Modality")
    # plt.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left')        

    # Add architecture labels
    for center, arch in zip(arch_centers, architectures):
        plt.text(center, -5, arch, ha='center', va='top', fontsize=11)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('subject dependent accuracies without labels.png', dpi=400, bbox_inches='tight')
    # plt.show()

def main():
    t0 = time.time()
    # subjects = ['subject_15', 'subject_16']     # CNN
    #subjects = ['subject_0']     # LSTM
    # subjects = ['subject_0', 'subject_1']     # Attention
    # subjects = ['subject_0', 'subject_1']
    
    sensor_name = 'EMG'
    # singleNet_save_path = 'real_time/SingleNet_CNN+LSTM+ATTENTION_EMG_complexModel_globalNorm_noWeight_7motions_lowpass/SingleNet_CNN+LSTM_EMG'
    singleNet_save_path = 'real_time/fine_tune/SingleNet_CNN+LSTM_EMG_newCrop_7motions_noLowpass/SingleNet_CNN+LSTM_EMG'
    singleNet_model_name = 'SingleNet_CNN_LSTM'

    fusionNet_save_path = 'subject_independent/FusionNet_LSTM_norm'  # noqa: F841
    fusionNet_model_name = 'FusionNet_LSTM'   # noqa: F841
    
    singleNet_classfication_real_time(subject_name = 'subject_0', sherpa_log_folder = singleNet_save_path, sensor_name = sensor_name, model_name = singleNet_model_name, num_motions = 7)
    
    # fusionNet_classfication_acrossSubjects(subject_name = subjects, sherpa_log_folder = fusionNet_save_path, model_name = fusionNet_model_name)

    # fusionNet_Kfold_classfication_independent(sherpa_log_folder = fusionNet_save_path, model_name = fusionNet_model_name, num_motions = 2)

    # # singleNet_Kfold_classfication_independent(sherpa_log_folder = singleNet_save_path, sensor_name = sensor_name, model_name = singleNet_model_name, num_motions = 2)
    # for path, mod in zip(['subject_independent/FusionNet_CNN+LSTM', 'subject_independent/FusionNet_LSTM'], ['FusionNet_CNN_LSTM', 'FusionNet_LSTM']):
    #     fusionNet_Kfold_classfication_independent(sherpa_log_folder = path, model_name = mod, num_motions = 2)
    
    print('Classification COMPLETE\n'
          'Time it took: ', time.time() - t0, 's')

def summary_accuracies():
    subjects = ['subject_0', 'subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11', 'subject_12', 'subject_13', 'subject_14', 'subject_15', 'subject_16']

    # Subject-dependent classification only for EEG
    accuracies = {
       
    "LSTM":{
        "EEG":[30.56, 33.33, 40.74, 33.33, 29.63, 34.85, 41.33, 64.20, 33.33, 33.33, 33.33, 33.33, 33.33, 35.90, 33.33, 30.77, 39.13],      
    },

    "CNN+LSTM":{
        "EEG":[70.83, 64.29, 61.73, 33.33, 35.80, 33.33, 45.33, 77.78, 33.33, 49.38, 52.38, 50.00, 46.15, 37.18, 26.39, 37.18, 47.83],
    },

    "CNN+LSTM+Attention":{
        "EEG":[77.78, 63.10, 62.96, 53.33, 50.62, 50.00, 58.67, 80.25, 44.87, 60.49, 57.14, 56.41, 50.00, 58.97, 45.83, 43.59, 50.72],   
    }}

    for key, val in accuracies.items():
        values = val['EEG']
        print(key)
        print('Mean: ', np.mean(values))
        print('STD: ', np.std(values))
        print()

    architectures = ['LSTM','CNN+LSTM','CNN+LSTM+Attention']
    _plot_subject_accuracy_hierarchical(subject_ids=subjects, accuracies=accuracies, architectures=architectures)

    # Subject-independent classification with LOSO for all subjects
    # accuracies = {
       
    # "LSTM":{
    #     "EEG":[],
    #     "EMG":[],           
    #     "Fusion":[]         
    # },

    # "CNN+LSTM":{
    #     "EEG":[],
    #     "EMG":[99.8],
    #     "Fusion":[99.7]
    # },

    # "CNN+LSTM+Attention":{
    #     "EEG":[48.2],             # (53.3) - higher Vloss
    #     "EMG":[99.4],
    #     "Fusion":[99.5]
    # }}

def compare_all_models():

    model_names = ['SingleNet_LSTM_EEG', 'SingleNet_LSTM_EMG']
    prediction_dict = {}
    for model in model_names:

        model_path_folder = Path(__file__).resolve().parent / f"loggings/{model}"
        sherpa_info_path = model_path_folder / 'subject_0-2/SHERPA_results.pt'

        if not os.path.exists(sherpa_info_path):
            raise FileExistsError(sherpa_info_path)
        
        data = torch.load(sherpa_info_path, weights_only=False)

        best_vloss = min(
            data["trials"],
            key=lambda x: x["validation_loss"]
        )

        prediction_dict[model] = best_vloss['predictions']
        print(best_vloss['labels'])
'''
def mcnemar_test(y_true, pred_A, pred_B):
    """
    Perform McNemar test between two models
    """

    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    # Contingency table
    n11 = np.sum((correct_A == 1) & (correct_B == 1))
    n10 = np.sum((correct_A == 1) & (correct_B == 0))
    n01 = np.sum((correct_A == 0) & (correct_B == 1))
    n00 = np.sum((correct_A == 0) & (correct_B == 0))

    table = [[n11, n10],
             [n01, n00]]

    # result = mcnemar(table, exact=False, correction=True)
    result = None

    return result.statistic, result.pvalue'''     

if __name__ == '__main__':
    # compare_all_models()
    
    # % MAXPOOL IS DEACTIVATED
    # COMPLEX MODEL ENABLED
    # Norm deactivated
    # main()
    
    # fusionNet_inspect_model(subject_name = 'all_subjects', sherpa_log_folder = 'subject_independent/FusionNet_LSTM_norm')
    singleNet_inspect_model(subject_name = 'all_subjects', sherpa_log_folder = 'SingleNet_CNN+LSTM_EMG', sensor_name = 'EMG', num_motions = 2)

    # singleNet_inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'subject_dependent/grip_app/SingleNet_CNN+LSTM+ATTENTION_EMG', num_motions=7)

    # for model in ['subject_dependent/SingleNet_LSTM_EEG','subject_dependent/SingleNet_CNN+LSTM_EEG','subject_dependent/SingleNet_CNN+LSTM+ATTENTION_EEG']:
    # for path in ['subject_independent/FusionNet_LSTM_norm', 'subject_independent/FusionNet_CNN+LSTM_norm', 'subject_independent/FusionNet_CNN+LSTM+ATTENTION']:
    # inspect_model(subject_name = 'subject_0', sherpa_log_folder = 'subject_dependent/grip_app/SingleNet_CNN+LSTM+ATTENTION_EMG', include_all=False)
    # real_time_inspect_model(subject_name = 'subject_0', 
    #                         sherpa_log_folder = 'SingleNet_CNN+LSTM_EMG_complexModel_globalNorm_noWeight_7motions_lowpass',
    #                         sensor_name = 'EMG',
    #                         num_motions = 7)
    # summary_accuracies()
    # subjects = [f'subject_{i}' for i in range(17)]
    # kfold_ins = KFoldManageDataset(None, None)
    # kfold_ins.create_kfold_splits_subject_independent(subject_ids=subjects, k=5)
    # pass


# {'learning_rate': 0.00015095576430059946, 'weight_decay': 0.0006720891424495656, 'dropout': 0.3118194976337473, 'batch_size': 128, 'dense_ratio': 0.75, 'activation': 'relu', 'num_hidden_units': 256, 'bidirectional': True, 'lstm_layers': 3, 'cnn_filters': 32, 'kernel_ratio': 7}


# {'input_dim': 3, 'output_dim': 15, 'hidden_dim': 256, 'lstm_layers': 3, 'bidirectional': True, 'dropout': 0.3118194976337473, 'activation': 'relu', 'dense_ratio': 0.75, 'cnn_filters': 32, 'kernel_size': 7}