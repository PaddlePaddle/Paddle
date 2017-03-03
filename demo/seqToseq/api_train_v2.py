import os

import paddle.v2 as paddle

from seqToseq_net_v2 import seqToseq_net_v2

# Data Definiation.
# TODO:This code should be merged to dataset package.
data_dir = "./data/pre-wmt14"
src_lang_dict = os.path.join(data_dir, 'src.dict')
trg_lang_dict = os.path.join(data_dir, 'trg.dict')

source_dict_dim = len(open(src_lang_dict, "r").readlines())
target_dict_dim = len(open(trg_lang_dict, "r").readlines())


def read_to_dict(dict_path):
    with open(dict_path, "r") as fin:
        out_dict = {
            line.strip(): line_count
            for line_count, line in enumerate(fin)
        }
    return out_dict


src_dict = read_to_dict(src_lang_dict)
trg_dict = read_to_dict(trg_lang_dict)

train_list = os.path.join(data_dir, 'train.list')
test_list = os.path.join(data_dir, 'test.list')

UNK_IDX = 2
START = "<s>"
END = "<e>"


def _get_ids(s, dictionary):
    words = s.strip().split()
    return [dictionary[START]] + \
           [dictionary.get(w, UNK_IDX) for w in words] + \
           [dictionary[END]]


def train_reader(file_name):
    def reader():
        with open(file_name, 'r') as f:
            for line_count, line in enumerate(f):
                line_split = line.strip().split('\t')
                if len(line_split) != 2:
                    continue
                src_seq = line_split[0]  # one source sequence
                src_ids = _get_ids(src_seq, src_dict)

                trg_seq = line_split[1]  # one target sequence
                trg_words = trg_seq.split()
                trg_ids = [trg_dict.get(w, UNK_IDX) for w in trg_words]

                # remove sequence whose length > 80 in training mode
                if len(src_ids) > 80 or len(trg_ids) > 80:
                    continue
                trg_ids_next = trg_ids + [trg_dict[END]]
                trg_ids = [trg_dict[START]] + trg_ids

                yield src_ids, trg_ids, trg_ids_next

    return reader


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    cost = seqToseq_net_v2(source_dict_dim, target_dict_dim)
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    reader_dict = {
        'source_language_word': 0,
        'target_language_word': 1,
        'target_language_next_word': 2
    }

    trn_reader = paddle.reader.batched(
        paddle.reader.shuffle(
            train_reader("data/pre-wmt14/train/train"), buf_size=8192),
        batch_size=5)

    trainer.train(
        reader=trn_reader,
        event_handler=event_handler,
        num_passes=10000,
        reader_dict=reader_dict)


if __name__ == '__main__':
    main()
