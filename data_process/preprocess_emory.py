import pandas as pd, numpy as np, pickle
from data_process.tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import constant
from collections import Counter

min_user_cont = 1000


def preprocess_text(x):
    # for punct in '"&}-/<>#$%\()*+:;=@[\\]^_`|\~':
    #     x = x.replace(punct, ' ')
    #
    # x = ' '.join(x.split())
    x = x.lower()

    return x


def create_utterances(filename, split):
    sentences, act_labels, emotion_labels, speakers, user_labels, conv_id, utt_id = [], [], [], [], [], [], []
    with open(filename, 'r') as f:
        data = json.load(f)
        episodes = data['episodes']
        for episode in episodes:
            scenes = episode['scenes']
            for scene in scenes:
                utterances = scene['utterances']
                c_id = scene['scene_id']
                sp_set = set([c['speakers'][0] for c in utterances])
                sp2idx = {sp: idx for idx, sp in enumerate(sp_set)}
                for utterance in utterances:
                    if utterance['transcript'].strip() == '':
                        utterance['utterance'] = 'noword'
                    sentences.append(utterance['transcript'])
                    emotion_labels.append(utterance['emotion'])
                    speakers.append(sp2idx[utterance['speakers'][0]])
                    act_labels.append('0')
                    conv_id.append(split[:2] + c_id)
                    utt_id.append(split[:2] + utterance['utterance_id'])
                    user_labels.append(utterance['speakers'][0])

    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))
    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id
    data['user_label'] = user_labels
    return data


def load_pretrained_glove():
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    f = open('/home/liuyijiang/resources/embed/glove.twitter.27B.200d.txt', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading pretrained GloVe model.")
    return glv_vector


def encode_labels(encoder, l, defult=None):
    if defult is None:
        return encoder[l]
    else:
        return encoder[l] if l in encoder else encoder[defult]


if __name__ == '__main__':
    dataset = 'emory'
    folder_path = f'../data/emory/'

    train_data = create_utterances(f'{folder_path}emotion-detection-train.json', 'train')
    valid_data = create_utterances(f'{folder_path}emotion-detection-dev.json', 'valid')
    test_data = create_utterances(f'{folder_path}emotion-detection-test.json', 'test')

    ## encode the emotion and dialog act labels ##
    all_act_labels, all_emotion_labels = set(train_data['act_label']), set(train_data['emotion_label'])
    all_user = train_data['user_label']
    all_user = Counter(all_user)
    all_user_labels = ['<pad>', '<unk>']
    for user, cont in all_user.items():
        if cont >= min_user_cont:
            all_user_labels.append(user)
    act_label_encoder, emotion_label_encoder, act_label_decoder, emotion_label_decoder = {}, {}, {}, {}
    user_label_encoder, user_label_decoder = {}, {}
    for i, label in enumerate(all_user_labels):
        user_label_encoder[label] = i
        user_label_decoder[i] = label

    for i, label in enumerate(all_act_labels):
        act_label_encoder[label] = i
        act_label_decoder[i] = label

    # for i, label in enumerate(all_emotion_labels):
    #     emotion_label_encoder[label] = i
    #     emotion_label_decoder[i] = label
    emotion_label_encoder = constant.EMOTION7
    emotion_label_decoder = {v: k for k, v in emotion_label_encoder.items()}
    pickle.dump(act_label_encoder, open(f'{folder_path}/act_label_encoder.pkl', 'wb'))
    pickle.dump(act_label_decoder, open(f'{folder_path}/act_label_decoder.pkl', 'wb'))
    pickle.dump(emotion_label_encoder, open(f'{folder_path}/emotion_label_encoder.pkl', 'wb'))
    pickle.dump(emotion_label_decoder, open(f'{folder_path}/emotion_label_decoder.pkl', 'wb'))
    pickle.dump(user_label_encoder, open(f'{folder_path}/user_label_encoder.pkl', 'wb'))
    pickle.dump(user_label_decoder, open(f'{folder_path}/user_label_decoder.pkl', 'wb'))
    print('act_label:', len(act_label_encoder))
    print('emotion_label:', len(emotion_label_encoder))
    print('user_label:', len(user_label_encoder))
    train_data['encoded_act_label'] = train_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))
    test_data['encoded_act_label'] = test_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))
    valid_data['encoded_act_label'] = valid_data['act_label'].map(lambda x: encode_labels(act_label_encoder, x))

    train_data['encoded_emotion_label'] = train_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))
    test_data['encoded_emotion_label'] = test_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))
    valid_data['encoded_emotion_label'] = valid_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))

    train_data['encoded_user_label'] = train_data['user_label'].map(
        lambda x: encode_labels(user_label_encoder, x, '<unk>'))
    test_data['encoded_user_label'] = test_data['user_label'].map(
        lambda x: encode_labels(user_label_encoder, x, '<unk>'))
    valid_data['encoded_user_label'] = valid_data['user_label'].map(
        lambda x: encode_labels(user_label_encoder, x, '<unk>'))

    ## tokenize all sentences ##
    all_text = list(train_data['sentence'])
    tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', oov_token='<unk>')
    tokenizer.fit_on_texts(all_text)
    pickle.dump(tokenizer, open(f'{folder_path}/tokenizer.pkl', 'wb'))

    ## convert the sentences into sequences ##
    train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence']))
    valid_sequence = tokenizer.texts_to_sequences(list(valid_data['sentence']))
    test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence']))

    train_data['sentence_length'] = [len(item) for item in train_sequence]
    valid_data['sentence_length'] = [len(item) for item in valid_sequence]
    test_data['sentence_length'] = [len(item) for item in test_sequence]

    max_num_tokens = 120
    train_sequence = pad_sequences(train_sequence, maxlen=max_num_tokens, padding='post')
    valid_sequence = pad_sequences(valid_sequence, maxlen=max_num_tokens, padding='post')
    test_sequence = pad_sequences(test_sequence, maxlen=max_num_tokens, padding='post')

    train_data['sequence'] = list(train_sequence)
    valid_data['sequence'] = list(valid_sequence)
    test_data['sequence'] = list(test_sequence)

    ## save the data in pickle format ##
    convSpeakers, convInputSequence, convInputMaxSequenceLength, convActLabels, convEmotionLabels, convUserLabel = {}, {}, {}, {}, {}, {}
    train_conv_ids, test_conv_ids, valid_conv_ids = set(train_data['conv_id']), set(test_data['conv_id']), set(
        valid_data['conv_id'])
    all_data = train_data.append(test_data, ignore_index=True).append(valid_data, ignore_index=True)

    print('Preparing dataset. Hang on...')
    for item in list(train_conv_ids) + list(test_conv_ids) + list(valid_conv_ids):
        df = all_data[all_data['conv_id'] == item]

        convSpeakers[item] = list(df['speaker'])
        convInputSequence[item] = list(df['sequence'])
        convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
        convActLabels[item] = list(df['encoded_act_label'])
        convEmotionLabels[item] = list(df['encoded_emotion_label'])
        convUserLabel[item] = list(df['encoded_user_label'])
    pickle.dump(
        [convSpeakers, convInputSequence, convInputMaxSequenceLength, convActLabels, convEmotionLabels, convUserLabel,
         train_conv_ids, test_conv_ids, valid_conv_ids], open(f'{folder_path}/daily_dialogue.pkl', 'wb'))

    ## save pretrained embedding matrix ##
    glv_vector = load_pretrained_glove()
    word_vector_length = len(glv_vector['the'])
    word_index = tokenizer.word_index
    inv_word_index = {v: k for k, v in word_index.items()}
    num_unique_words = len(word_index)
    glv_embedding_matrix = np.zeros((num_unique_words + 1, word_vector_length))

    for j in range(1, num_unique_words + 1):
        try:
            glv_embedding_matrix[j] = glv_vector[inv_word_index[j]]
        except KeyError:
            glv_embedding_matrix[j] = np.random.randn(word_vector_length) / 200

    np.ndarray.dump(glv_embedding_matrix, open(f'{folder_path}/glv_embedding_matrix', 'wb'))
    print('Done. Completed preprocessing.')
