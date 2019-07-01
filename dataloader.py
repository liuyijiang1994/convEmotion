import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from torch.utils.data import DataLoader


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.Users, \
        self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.LongTensor(self.InputSequence[conv]), \
               torch.LongTensor(self.Speakers[conv]), \
               torch.FloatTensor([1] * len(self.ActLabels[conv])), \
               torch.LongTensor(self.ActLabels[conv]), \
               torch.LongTensor(self.Users[conv]), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               self.InputMaxSequenceLength[conv], \
               conv

    def __len__(self):
        return self.len


class DailyDialoguePadCollate:

    def __init__(self, user_size, dim=0):
        self.dim = dim
        self.user_size = user_size

    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))

        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]

        # stack all
        return torch.stack(batch, dim=0)

    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        batch_speaker = dat[4]
        spk_t = []
        for spk in batch_speaker:
            t = torch.zeros(len(spk), self.user_size)
            for idx, indice in enumerate(spk):
                t[idx][indice] = 1
            spk_t.append(t)
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i == 0 else \
                    pad_sequence(spk_t) if i == 1 else \
                        pad_sequence(dat[i], True) if i < 5 else \
                            pad_sequence(dat[i], True, padding_value=-1) if i == 5 else \
                                dat[i].tolist() for i in dat]
