import glob
import os
import numpy as np
import pandas as pd
from collections import Counter
import random

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as torchaudio_F
import torchaudio.transforms as torchaudio_T
from torchaudio.backend.soundfile_backend import load # https://pytorch.org/audio/stable/backend.html#load


########### For classification task ###########
def build_dataset(wav_path, dataset_path, top_labels=None, numeric_only=True): 
    path_list = glob.glob(wav_path+'/*.wav')
    dialog_speaker_id = list(map(lambda x: x.split('/')[-1].split('.')[0], path_list))
    dialog_id = list(map(lambda x: x.split('/')[-1].split('_')[0], path_list))
    df_path = pd.DataFrame({'dialog_id': dialog_id, 'dialog_speaker_id': dialog_speaker_id, 'path': path_list})

    df_metadata = pd.read_csv(dataset_path)
    df_metadata['dialog_speaker_id'] = df_metadata['dialog_id'] + '_' + df_metadata['speaker']
    df_metadata = df_metadata.merge(df_path, how='inner')

    # Compute Top 10 tags for train
    if top_labels == None:
        cnt = Counter(df_metadata['da_tag'])
        print(cnt.most_common()[0:10])
        top_labels = list(map(lambda x: x[0], cnt.most_common()[0:10]))
    label_map = dict(zip(top_labels, range(10)))

    df_metadata = df_metadata[df_metadata['da_tag'].apply(lambda x: x in top_labels)]
    df_metadata['label'] = df_metadata['da_tag'].apply(lambda x: label_map[x])

    df_metadata.sort_values(['dialog_id', 'speaker', 'start_time'], inplace=True)
    df_metadata = df_metadata.reset_index(drop=True)
    if numeric_only:
        df_metadata = df_metadata.drop(['dialog_speaker_id', 'path'], axis=1)
    
    return df_metadata, top_labels


def compute_sample_id(path): 
    df_data = pd.read_csv(path)[['dialog_id', 'speaker', 'da_tag', 'start_time', 'end_time']]
    df_collect = []

    dialog_id_arr = df_data['dialog_id'].unique()
    for dialog_id in dialog_id_arr:
        for speaker in ['A', 'B']:
            df_target = df_data[(df_data['dialog_id'] == dialog_id) & (df_data['speaker'] == speaker)].reset_index(drop=True).copy()
            sample_id = list(range(df_target.shape[0]))
            sample_id = list(map(lambda x: dialog_id + '_' + speaker + '_' + str(x), sample_id))
            df_target['sample_id'] = sample_id
            df_collect.append(df_target)

    df_sample_id = pd.concat(df_collect).reset_index(drop=True)
    
    return df_sample_id
    

########## For LibriSpeech ##########
# Reference: https://github.com/domerin0/rnn-speech/tree/master
def find_files(root_search_path, files_extension):
    files_list = []
    for root, _, files in os.walk(root_search_path):
        files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
    return files_list


def clean_label(_str):
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    _str = _str.strip()
    _str = _str.lower()
    _str = _str.replace(".", "")
    _str = _str.replace(",", "")
    _str = _str.replace("?", "")
    _str = _str.replace("!", "")
    _str = _str.replace(":", "")
    _str = _str.replace("-", " ")
    _str = _str.replace("_", " ")
    _str = _str.replace("  ", " ")
    return _str


def get_data_librispeech(raw_data_path):
    text_files = find_files(raw_data_path, ".txt")
    result = []
    for text_file in text_files:
        directory = os.path.dirname(text_file)
        with open(text_file, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                head = line.split(' ')[0]
                if len(head) < 5:
                    # Not a line with a file desc
                    break
                audio_file = directory + "/" + head + ".flac"
                if os.path.exists(audio_file):
                    result.append([audio_file, clean_label(line.replace(head, "")), None])
    
    return result


class LibriSpeech_Dataset(Dataset):
    def __init__(self, base_path, sample_rate=16000, resample_rate = 8000, t_ceiling = 20, test=False):
        self.base_path = base_path
        self.metadata = get_data_librispeech(base_path)
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.t_ceiling = t_ceiling
        self.n_ceiling = sample_rate*t_ceiling
        self.resampler = torchaudio_T.Resample(sample_rate, resample_rate)
        
        if test:
            self.metadata = self.metadata[0:31]

    def __getitem__(self, i):
        path = self.metadata[i][0]
        waveform, sample_rate = torchaudio.load(path)
        resampled_waveform = self.resampler(waveform[:, :self.n_ceiling])
        assert(sample_rate == self.sample_rate)
        
        return resampled_waveform

    def __len__(self):
        return len(self.metadata)
    
    
# Reference: https://stackoverflow.com/questions/55041080/how-does-pytorch-dataloader-handle-variable-size-data
def padding_tensor(sequences):
    batch_size = len(sequences) # dim 0
    feature_dim = sequences[0].shape[0] # dim 1
    max_len = max([s.size(1) for s in sequences]) # get max length in dim 2

    out_dims = (batch_size, feature_dim, max_len)
    out_tensor = torch.zeros(out_dims)

    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :, :length] = tensor # fill dim=i
        
    return out_tensor


########## For Switchboard DA ##########
# get 5 second audio before end_time
class Switchboard_Dataset_v1(Dataset):
    def __init__(self, df_metadata, sample_rate=8000):
        self.sample_rate = sample_rate
        self.df_metadata = df_metadata

    def __getitem__(self, i):
        row = self.df_metadata.loc[i]
        end_idx = int(row['end_time']*self.sample_rate)
        start_idx = max(0, end_idx - self.sample_rate*5)
        
        waveform, _ = load(row['path'], frame_offset=start_idx, num_frames=self.sample_rate*5)
        
        return waveform

    def __len__(self):
        return len(self.df_metadata)
    
# return: 1. padding each audio into same length; 2. idx_end
def padding_tensor_extractor_v1(sequences):
    batch_size = len(sequences) # dim 0
    feature_dim = sequences[0].shape[0] # dim 1
    max_len = max([s.size(1) for s in sequences]) # get max length in dim 2

    x = torch.zeros((batch_size, feature_dim, max_len))
    x_len = np.zeros(batch_size)

    for i, x_i in enumerate(sequences):
        x_i_len = x_i.size(1)
        x[i, :, :x_i_len] = x_i
        
        x_len[i] = x_i_len
        
    return x, x_len.astype(int)


# get 5 second audio before end_time and 3 second audio after end_time
class Switchboard_Dataset_trainer_v3(Dataset):
    def __init__(self, df_metadata, sample_rate=8000):
        self.sample_rate = sample_rate
        self.df_metadata = df_metadata

    def __getitem__(self, i):        
        sample_rate = self.sample_rate
        row = self.df_metadata.loc[i]
        start_t_idx = int(row['start_time']*sample_rate)
        end_t_idx = int(row['end_time']*sample_rate)
        pre_window = sample_rate*5
        post_window = sample_rate*3
        start_idx = max(0, end_t_idx-pre_window)

        waveform, _ = load(row['path'], frame_offset=start_idx, num_frames=pre_window+post_window)

        delta1 = max(0, start_t_idx - start_idx)
        delta2 = end_t_idx - max(start_t_idx, start_idx)

        indicator_tensor = torch.zeros(waveform.shape)
        indicator_tensor[0,delta1:delta1+delta2] = 1

        output_tensor = torch.concat((waveform, indicator_tensor))
        
        return output_tensor, row['label']

    def __len__(self):
        return len(self.df_metadata)
    
    
# return: 1. padding each audio into same length; 2. label
def padding_tensor_trainer_v3(sequences):
    batch_size = len(sequences) # dim 0
    feature_dim = sequences[0][0].shape[0] # dim 1
    max_len = max([s[0].size(1) for s in sequences]) # get max length in dim 2

    x_dims = (batch_size, feature_dim, max_len)
    x = torch.zeros(x_dims)
    
    x_len = np.zeros(batch_size)
    y = np.zeros(batch_size)

    for i, tuple_in in enumerate(sequences):
        x_i = tuple_in[0]
        x_length = x_i.size(1)
        x[i, :, :x_length] = x_i # fill dim=i
        
        x_len[i] = x_length
        y[i] = tuple_in[1]
        
    return x, torch.tensor(y.astype(int))

