import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

tokenizer_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/gpt2"


class MTDataset(Dataset):
    def __init__(self, text_path, motion_path, tokenizer):
        self.data = load_from_dir(text_path, motion_path, tokenizer)
        self.inp_PAD = 0
        self.tgt_PAD = 1024
        self.tgt_BOS = 1025
        self.tgt_EOS = 1026

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        tgt_tokens = [[self.tgt_BOS] + sent + [self.tgt_EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_text],
                                   batch_first=True, padding_value=self.inp_PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.tgt_PAD)
        
        batch_input_mask = batch_input != self.inp_PAD
        batch_target_mask = batch_target != self.tgt_PAD

        return {"src":batch_input, "tgt":batch_target, "src_mask":batch_input_mask, "tgt_mask":batch_target_mask}


def load_from_dir(text_path, motion_path, tokenizer):
    text_list = []
    for filepath,dirnames,filenames in os.walk(text_path):
        for filename in filenames:
            with open(os.path.join(filepath,filename), 'rb') as f:
                raw_text = pickle.load(f)[0]['text']
                text = []
                for sent in raw_text:
                    encoded_sent = tokenizer.encode(sent)
                    text.append(encoded_sent)
                    # print(encoded_sent.shape, model(encoded_sent)['last_hidden_state'].shape)
                text_list.append(text)
    motion_list = []
    for filepath,dirnames,filenames in os.walk(motion_path):
        for filename in filenames:
            with open(os.path.join(filepath,filename), 'rb') as f:
                motion = pickle.load(f)
                motion_list.append(motion.tolist())
    data = []
    for i in range(len(text_list)):
        for text in text_list[i]:
            data.append([text, motion_list[i]])
    return data
