import os, sys
import struct
import numpy as np
from paddle.v2.data import IDataIter

word_dict = dict()  # load or build dict
label_dict = dict()  # load or build dict
predicate_dict = dict()  # load or build dict


def load_data(filename='./data/feature'):
    data = dict()
    data['word_slot'] = []
    data['ctx_n2_slot'] = []
    data['ctx_n1_slot'] = []
    data['ctx_0_slot'] = []
    data['ctx_p1_slot'] = []
    data['ctx_p2_slot'] = []
    data['predicate_slot'] = []
    data['mark_slot'] = []
    data['label_slot'] = []

    with open(filename, "rb") as fn:
        for line in fdata:
            sentence, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2,  mark, label = \
                line.strip().split('\t')

            words = sentence.split()
            sen_len = len(words)
            word_slot = [word_dict.get(w, UNK_IDX) for w in words]

            predicate_slot = [predicate_dict.get(predicate)] * sen_len
            ctx_n2_slot = [word_dict.get(ctx_n2, UNK_IDX)] * sen_len
            ctx_n1_slot = [word_dict.get(ctx_n1, UNK_IDX)] * sen_len
            ctx_0_slot = [word_dict.get(ctx_0, UNK_IDX)] * sen_len
            ctx_p1_slot = [word_dict.get(ctx_p1, UNK_IDX)] * sen_len
            ctx_p2_slot = [word_dict.get(ctx_p2, UNK_IDX)] * sen_len

            marks = mark.split()
            mark_slot = [int(w) for w in marks]

            label_list = label.split()
            label_slot = [label_dict.get(w) for w in label_list]

            data['word_slot'].append(word_slot)
            data['ctx_n2_slot'].append(ctx_n2_slot)
            data['ctx_n1_slot'].append(ctx_n1_slot)
            data['ctx_0_slot'].append(ctx_0_slot)
            data['ctx_p1_slot'].append(ctx_p1_slot)
            data['ctx_p2_slot'].append(ctx_p2_slot)
            data['mark_slot'].append(mark_slot)
            data['label_slot'].append(label_slot)

    return data


class Reader(IDataIter):
    def __init__(self, data, batch_size, is_shuffle=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.index_in_epoch = 0
        self.data = data
        self.num_examples = len(self.data[self.data.keys()[0]])

    def __iter__(self):
        def shuffle():
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            for k in self.data.keys():
                self.data[k] = self.data[k][perm]

        if self.is_shuffle:
            shuffle()
        return self

    def next(self):
        if self.index_in_epoch >= self.num_examples:
            self.index_in_epoch = 0
            raise StopIteration

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        end = min(self.index_in_epoch, self.num_examples)
        ret_val = dict()
        for k in self.data.keys():
            ret_val[k] = self.data[k][start:end]
        return ret_val


def main():
    data = load_data('./data/feature')
    train_data = Reader(data, 128, True)

    for i in xrange(2):
        print '---start pass---'
        for data_batch in train_data:
            print data_batch['pixel']


if __name__ == "__main__":
    main()
