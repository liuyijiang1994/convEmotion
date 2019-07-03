from train import get_DailyDialogue_loaders, process_data_loader
import pickle
from collections import Counter


def get_label_weight(train_loader):
    label_list = []
    for data in train_loader:
        input_sequence, qmask, umasks, users, label = process_data_loader(data)
        '''
                input_sequence: sent_num, batch_sz, max_seq_length
                         qmask: sent_num, batch_sz, party
                         umask: batch_sz, sent_num
                         label: batch_sz, sent_num
                         users: batch_sz, sent_num
        '''
        label = label.cpu().numpy()
        for l in label:
            l = l.tolist()
            label_list.extend(l)
    label_counter = Counter(label_list)
    conts = [label_counter.get(i) for i in range(7)]
    weight = [1000 / cont for cont in conts]
    return weight
