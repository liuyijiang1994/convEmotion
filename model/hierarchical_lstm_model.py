import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.SubLayers import MultiHeadAttention
from utils import get_seq_len_batch
import constant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constant.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(constant.PAD).type(torch.float).unsqueeze(-1)


class Hierarchichal_LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_lstm_hidden, sent_lstm_hidden, n_classes=7, dropout=0.5):
        super(Hierarchichal_LSTM_Model, self).__init__()
        self.feature_extractor = LSTMFeatureExtractor(vocab_size, embedding_dim, word_lstm_hidden)
        self.sent_lstm = nn.LSTM(input_size=word_lstm_hidden * 2,
                                 hidden_size=sent_lstm_hidden,
                                 bias=True,
                                 batch_first=True,
                                 bidirectional=True)
        self.proj = nn.Linear(sent_lstm_hidden * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, utterance, user_label, umask):
        '''
        :param x: utt_sequence [num_utt, batch,num_words]
        :param umask: utt_mask [num_utt, batch]
        :return:
        '''
        sent_representation = self.feature_extractor(utterance, umask)
        batch_sz, seq_num, _ = sent_representation.shape
        indices, desorted_indices = sort_by_seq_len(user_label, 0)
        sent_representation = sent_representation[indices]
        user_label = user_label[indices]
        mask = user_label.data.gt(0).float()
        x = nn.utils.rnn.pack_padded_sequence(sent_representation, mask.sum(1).int(), batch_first=True)
        h, _ = self.sent_lstm(x)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=seq_num)
        h = h[desorted_indices]
        out = self.dropout(h)
        out = self.proj(out)
        return out, [], [], []

    def init_pretrained_embeddings_from_numpy(self, embed):
        self.feature_extractor.init_pretrained_embeddings_from_numpy(embed)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, dropout=0.5):
        super(LSTMFeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_sz = output_size
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=output_size,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)

        self.slf_attn = MultiHeadAttention(1, output_size * 2, output_size * 2, output_size * 2, dropout=dropout)

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        self.embedding.weight.requires_grad = True

    def forward(self, x, umask):
        '''
        :param x: utt_sequence [num_utt, batch,num_words]
        :param umask: utt_mask [num_utt, batch]
        :return:# (num_utt, batch, output_size) -> (num_utt, batch, output_size)
        '''
        x = x.transpose(0, 1).contiguous()  # [batch_num, num_utt, num_words]
        batch_num, seq_num, seq_len = x.size()
        word_mask = x != 0
        contexts = x.view(batch_num * seq_num, -1)  # batch * sen_num, num_words
        contexts = self.embedding(contexts)  # batch * sen_num, seqlen, embed_sz
        contexts = contexts.view(batch_num, seq_num, seq_len, -1)  # batch, sen_num, seqlen, embed_sz

        batch_sent_num = umask.sum(dim=1).int()
        sent_h = []
        for idx, b in enumerate(contexts):
            b_batch_sz, b_seq_len, _ = b.shape
            _word_mask = word_mask[idx]
            b = b[:batch_sent_num[idx]]
            ori_x_b = x[idx]
            ori_x_b = ori_x_b[:batch_sent_num[idx]]
            _word_mask = _word_mask[:batch_sent_num[idx]]

            indices, desorted_indices = sort_by_seq_len(_word_mask, 0)
            mask = _word_mask.data.gt(0).float()  # 获得input的mask（填充位置为0）
            b = b[indices]
            mask = mask[indices]

            b = nn.utils.rnn.pack_padded_sequence(b, mask.sum(1).int(), batch_first=True)
            h, _ = self.encoder(b)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=seq_len)
            h = h[desorted_indices]
            # pos = self.position_enc(x_pos_b)
            # h = h + pos

            # slf_attn_mask = get_attn_key_pad_mask(seq_k=ori_x_b, seq_q=ori_x_b)
            # non_pad_mask = get_non_pad_mask(ori_x_b)
            #
            # enc_output, enc_slf_attn = self.slf_attn(
            #     h, h, h, mask=slf_attn_mask)
            # enc_output *= non_pad_mask
            enc_output = h

            enc_output = enc_output.transpose(2, 1)  # batch x hidden x seqlen
            enc_output = F.max_pool1d(enc_output, enc_output.shape[2]).squeeze(2)

            if batch_sent_num[idx] < seq_num:
                pad = torch.zeros([seq_num - batch_sent_num[idx], self.output_sz * 2]).to(device)
                enc_output = torch.cat((enc_output, pad), dim=0)
            sent_h.append(enc_output)

        sent_h = torch.stack(sent_h, dim=0)  # batch, sent_num, hidden
        return sent_h


def sort_by_seq_len(data, pad):
    # 对text 和 tag 按seq_len排序，便于使用pack_padded_sequence
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    seq_lengths = get_seq_len_batch(data, pad)
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)
    return indices, desorted_indices
