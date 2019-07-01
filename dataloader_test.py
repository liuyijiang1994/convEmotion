from train import get_DailyDialogue_loaders, process_data_loader
import pickle
import torch

data_folder = f'data/emory/'
batch_size = 2
user_label_encoder = pickle.load(open(f'{data_folder}speaker_label_encoder.pkl', 'rb'))
user_size = len(user_label_encoder)
train_loader, valid_loader, test_loader = get_DailyDialogue_loaders(f'{data_folder}daily_dialogue.pkl', user_size,
                                                                    batch_size=batch_size, num_workers=0)
tokenizer = pickle.load(open(f'{data_folder}/tokenizer.pkl', 'rb'))
emotion_label_decoder = pickle.load(open(f'{data_folder}/emotion_label_decoder.pkl', 'rb'))

for data in train_loader:
    input_sequence, qmask, umasks, users, label = process_data_loader(data)
    '''
            input_sequence: sent_num, batch_sz, max_seq_length
                     qmask: sent_num, batch_sz, party
                     umask: batch_sz, sent_num
                     label: batch_sz, sent_num
                     users: batch_sz, sent_num
    '''
    for i in input_sequence, qmask, umasks, users, label:
        print(i.shape)
    break
    print(qmask.shape)
    input_sequence = input_sequence.transpose(0, 1)
    # qmask = qmask.transpose(0, 1)
    # for texts, emotions in zip(input_sequence, label):
    #     texts = texts.cpu().numpy()
    #     texts = tokenizer.sequences_to_texts(texts)
    #     for text, emotion in zip(texts, emotions):
    #         if emotion.cpu().item() != -1:
    #             print(text, emotion_label_decoder.get(emotion.cpu().item(), 'pad'))
    #     print('-' * 32)
    for texts, umask in zip(input_sequence, umasks):
        for t, u in zip(texts, umask):
            t_value = t.sum().cpu().item()
            u_value = u.cpu().item()
            if t_value == 0 and u_value != 0:
                print(t, u)
