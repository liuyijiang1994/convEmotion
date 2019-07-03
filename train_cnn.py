import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import numpy as np, pickle, time, argparse
from dataloader import DailyDialoguePadCollate, DailyDialogueDataset
import adabound
from model.cnn_model import CNNFeatureExtractor
from utils import weight_init
from data_process.data_helper import get_label_weight
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support

torch.manual_seed(233)
data_folder = 'data/emory/'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


def get_DailyDialogue_loaders(path, user_size, batch_size=32, num_workers=0, pin_memory=False):
    trainset = DailyDialogueDataset('train', path)
    testset = DailyDialogueDataset('test', path)
    validset = DailyDialogueDataset('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=DailyDialoguePadCollate(user_size, dim=0),
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=DailyDialoguePadCollate(user_size, dim=0),
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=DailyDialoguePadCollate(user_size, dim=0),
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def process_data_loader(data):
    '''
       input_sequence, qmask,      umask,       act_labels, emotion_labels, max_sequence_lengths, _
       输入序列，    ，speaker[0,1]， actlabel[1], actlableid, emotionid,      最大长度，             第几个
       '''
    input_sequence, qmask, umask, act_labels, user_labels, emotion_labels, max_sequence_lengths, _ = data
    input_sequence = input_sequence[:, :, :max(max_sequence_lengths)]

    input_sequence, qmask, umask, user_labels = input_sequence.to(device), qmask.to(device), \
                                                umask.to(device), user_labels.to(device)
    # act_labels = act_labels.to(device)
    emotion_labels = emotion_labels.to(device)

    return [input_sequence, qmask, umask, user_labels, emotion_labels]


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    loss_all = 0

    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        input_sequence, qmask, umask, user_labels, label = process_data_loader(data)
        '''
            input_sequence: sent_num, batch_sz, max_seq_length
                     qmask: sent_num, batch_sz, party
                     umask: batch_sz, sent_num
               user_labels: batch_sz, sent_num
                     label: batch_sz, sent_num
        '''
        log_prob, alpha, alpha_f, alpha_b = model(input_sequence, umask)
        # log_prob: batch,sent_num,class
        lp_ = log_prob.view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_)
        # if train:
        #     print(loss.item())
        loss_all += loss.item()
        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas.extend(alpha)
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    uwa, acc, f1, acc_map = utils.evaluate(preds, labels)
    return loss_all / len(dataloader), uwa, acc, f1, labels, preds, masks, [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=500, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')

    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes = 7
    cuda = args.cuda
    n_epochs = args.epochs

    feat_output_size = 128
    D_m = feat_output_size * 2
    D_g = 256
    D_p = 256
    D_e = 256
    D_h = 256
    D_a = 128
    hidden_sz = 128
    dropout = 0.5
    PATIENCE_CONSTANT = 15
    patience = PATIENCE_CONSTANT

    glv_pretrained = np.load(open(f'{data_folder}glv_embedding_matrix', 'rb'))
    vocab_size, embedding_dim = glv_pretrained.shape

    user_label_encoder = pickle.load(open(f'{data_folder}user_label_encoder.pkl', 'rb'))
    user_size = len(user_label_encoder)
    model = CNNFeatureExtractor(vocab_size=vocab_size,
                                embedding_dim=embedding_dim,
                                output_size=hidden_sz,
                                filters=128,
                                kernel_sizes=[2, 3, 4],
                                n_classes=7,
                                dropout=0.5)
    model.apply(weight_init)
    model.init_pretrained_embeddings_from_numpy(glv_pretrained)
    if cuda:
        model.to(device)
    train_loader, valid_loader, test_loader = get_DailyDialogue_loaders(f'{data_folder}daily_dialogue.pkl',
                                                                        user_size=user_size,
                                                                        batch_size=batch_size, num_workers=0)
    loss_weights = torch.FloatTensor(get_label_weight(train_loader)).to(device)
    # loss_weights = torch.FloatTensor(report('data/EmotionPush/emotionpush_train.json')).to(device)
    print('loss_weights', loss_weights)
    loss_function = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=-1)
    # loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=args.lr,
    #                        weight_decay=args.l2)
    optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, final_lr=0.1,
                                  weight_decay=1e-4)

    best_loss, best_label, best_pred, best_mask = 999, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_uwa, train_acc, train_f1, _, _, _, _ = train_or_eval_model(model, loss_function,
                                                                                     train_loader, e, optimizer, True)
        valid_loss, valid_uwa, valid_acc, valid_f1, valid_label, valid_pred, valid_mask, valid_attentions = train_or_eval_model(
            model, loss_function,
            valid_loader, e)
        test_loss, test_uwa, test_acc, test_f1, test_label, test_pred, test_mask, test_attentions = train_or_eval_model(
            model, loss_function,
            test_loader, e)

        if best_loss > valid_loss:
            best_valid_loss, best_valid_label, best_valid_pred, best_valid_mask, best_valid_attn = valid_loss, valid_label, valid_pred, valid_mask, valid_attentions
            best_test_loss, best_test_label, best_test_pred, best_test_mask, best_test_attn = test_loss, test_label, test_pred, test_mask, test_attentions
            best_loss = valid_loss
            patience = PATIENCE_CONSTANT
            print("[!] saving model...")
            if not os.path.isdir("checkpoint/save"):
                os.makedirs("checkpoint/save")
            else:
                torch.save(model.state_dict(), 'checkpoint/save/model.pt')
        else:
            patience -= 1

        if args.tensorboard:
            writer.add_scalar('valid: accuracy/loss', valid_acc / valid_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
        print(
            'epoch {} train_loss {} train_uwa {} train_acc {} train_fscore {} time {}s'.format(
                e + 1, train_loss, train_uwa, train_acc, train_f1, round(time.time() - start_time, 2)))
        print(
            'epoch {} valid_loss {} valid_uwa {} valid_acc {} valid_fscore {} time {}s'.format(
                e + 1, valid_loss, valid_uwa, valid_acc, valid_f1, round(time.time() - start_time, 2)))
        print(
            'epoch {} test_loss {} test_uwa {} test_acc {} test_fscore {} time {}s'.format(
                e + 1, test_loss, test_uwa, test_acc, test_f1, round(time.time() - start_time, 2)))
        print('patience:', patience)
        if patience == 0:
            break
    if args.tensorboard:
        writer.close()

    utils.evaluate(best_test_pred, best_test_label, show=True)
    print(confusion_matrix(best_test_label, best_test_pred))
