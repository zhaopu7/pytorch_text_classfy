# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import data_polarity as reader
import random
import time


class Net(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, filter_num, hidden_size):
        super(Net, self).__init__()

        def get_conv(filter_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filter_num,  # filters
                    kernel_size=(filter_size, emb_dim),  # (height, width)
                    stride=1
                ),
                nn.Tanh(),
                nn.MaxPool2d((seq_len - filter_size + 1, 1))
            )

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.conv2 = get_conv(2, filter_num)
        self.conv3 = get_conv(3, filter_num)
        self.conv4 = get_conv(4, filter_num)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + filter_num * 3, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        # x : [batch_size, seq_len]
        x_emb = self.emb(x)
        [batch_size, seq_len, emb_dim] = x_emb.size()
        conv_input = x_emb.view(batch_size, 1, seq_len, emb_dim)  # [batch_size, channel, seq_len, emb_dim]
        conv2_encode = self.conv2(conv_input).view(batch_size, -1)  # [batch_size, filter_num]
        conv3_encode = self.conv3(conv_input).view(batch_size, -1)  # [batch_size, filter_num]
        conv4_encode = self.conv4(conv_input).view(batch_size, -1)  # [batch_size, filter_num]
        _, (rnn_encode, _) = self.rnn(x_emb, None)
        encode = torch.cat([rnn_encode[0], rnn_encode[1], conv2_encode, conv3_encode, conv4_encode],
                           dim=1)  # [batch_size, hidden_size * 2 + filter_num * 3]
        x = self.fc(encode)
        output = self.out(x)
        return output


def train():
    vocab_file = 'rt-polaritydata/vocab.txt'
    data_dir = 'rt-polaritydata/rt-polaritydata/'
    batch_size = 32
    time_step = 30
    eval_per_iter = 100
    vocab = reader.load_vocab(vocab_file)
    x, x_len, y = reader.read_data(data_dir, vocab, seq_len=time_step)
    sample_num = len(x)

    net = Net(vocab_size=len(vocab), emb_dim=5, seq_len=time_step, filter_num=2, hidden_size=10)
    print net
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # optimize all cnn parameters
    loss_func = nn.BCELoss()  # the target label is not one-hotted

    iter = 0
    while True:
        """train one batch"""
        iter += 1
        x_batch, x_len_batch, y_batch = [], [], []
        for _ in range(batch_size):
            idx = random.randint(0, sample_num - 1)
            x_batch.append(x[idx])
            x_len_batch.append(x_len[idx])
            y_batch.append(y[idx])
        st = time.time()
        x_batch = Variable(torch.LongTensor(x_batch))
        y_batch_pred = net(x_batch)
        loss = loss_func(y_batch_pred, Variable(torch.FloatTensor(y_batch)).view(batch_size, 1))
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        """test on current batch"""
        if iter % eval_per_iter == 0:
            right_count = 0.0
            y_batch_pred = y_batch_pred.data.numpy()
            for i in range(batch_size):
                if y_batch_pred[i][0] > 0.5 and y_batch[i] == 1:
                    right_count += 1
                elif y_batch_pred[i][0] < 0.5 and y_batch[i] == 0:
                    right_count += 1

            print "iter: %d, acc: %.4f, time cost this iter: %.4f" % (iter, right_count / batch_size,
                                                                     time.time() - st)

            """save"""
            """
            torch.save(net, 'net.pkl')  # save entire net
            net = torch.load('net.pkl') # load
            
            torch.save(net.state_dict(), 'net_params.pkl')  # save only the parameters
            net.load_state_dict(torch.load('net_params.pkl')) # load
            """

if __name__ == '__main__':
    train()
