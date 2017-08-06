# coding=utf-8
import os


def build_vocab(dir, vocab_file, vocab_size):
    wc = {}
    for fn in os.listdir(dir):
        with open(dir + fn) as f:
            for line in f:
                for w in line.strip().split():
                    if w in wc:
                        wc[w] += 1
                    else:
                        wc[w] = 1
    wc = sorted(wc.items(), key=lambda d: d[1], reverse=True)
    f = open(vocab_file, 'w')
    f.write('UNK\nEOS\n')
    for i in range(vocab_size - 2):
        f.write(wc[i][0] + '\n')
    f.close()
    print 'build vocab finished.'


def load_vocab(vocab_file):
    vocab = {}
    id = 0
    with open(vocab_file) as f:
        for line in f:
            vocab[line.strip()] = id
            id += 1
    print 'vocab size:', len(vocab)
    return vocab


def read_data(dir, vocab, seq_len):
    def sentence_to_ids(sent):
        return [vocab.get(w, vocab['UNK']) for w in sent.split()]

    x = []
    x_len = []
    lables = []
    for fn in os.listdir(dir):
        with open(dir + fn) as f:
            lable = 1
            if 'neg' in fn:
                lable = 0
            for line in f:
                line = line.strip().split()
                length = len(line)
                if length > seq_len:
                    continue
                ids = [vocab['EOS']] * seq_len
                ids[0:length] = [vocab.get(w, vocab['UNK']) for w in line]
                x.append(ids)
                x_len.append(length)
                lables.append(lable)
    return x, x_len, lables


if __name__ == '__main__':
    vocab_file = 'rt-polaritydata/vocab.txt'
    data_dir = 'rt-polaritydata/rt-polaritydata/'

    # build_vocab(data_dir, vocab_file, 3000)

    vocab = load_vocab(vocab_file)
    read_data(data_dir, vocab, 30)
