import os, sys
import struct
import numpy as np

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


def reader(data, batch_size, is_shuffle=True):
    def make_shuffle(data, label):
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        for k in self.data.keys():
            self.data[k] = self.data[k][perm]

    def make_minibatch(data):
        if is_shuffle:
            make_shuffle(data, label)

        num_sample = len(self.data.keys()[0])
        for start in xrange(0, num_sample, batch_size):
            end = min(start + batch_size, len(data))
            ret_val = dict()
            for k in self.data.keys():
                ret_val[k] = self.data[k][start:end]
            yield ret_val

    return make_minibatch(data)


def main():
    data = load_data('./data/feature')

    for i in xrange(2):
        print '---start pass---'
        for data_batch in reader(data, 128, True):
            print data_batch['pixel']


if __name__ == "__main__":
    main()
