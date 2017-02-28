import numpy
import paddle.v2 as paddle
from model_v2 import db_lstm

UNK_IDX = 0

word_dict_file = './data/wordDict.txt'
label_dict_file = './data/targetDict.txt'
predicate_file = './data/verbDict.txt'

word_dict = dict()
label_dict = dict()
predicate_dict = dict()

with open(word_dict_file, 'r') as f_word, \
     open(label_dict_file, 'r') as f_label, \
     open(predicate_file, 'r') as f_pre:
    for i, line in enumerate(f_word):
        w = line.strip()
        word_dict[w] = i

    for i, line in enumerate(f_label):
        w = line.strip()
        label_dict[w] = i

    for i, line in enumerate(f_pre):
        w = line.strip()
        predicate_dict[w] = i

word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_len = len(predicate_dict)

print 'word_dict_len=%d' % word_dict_len
print 'label_dict_len=%d' % label_dict_len
print 'pred_len=%d' % pred_len


def train_reader(file_name="data/feature"):
    def reader():
        with open(file_name, 'r') as fdata:
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
                yield word_slot, ctx_n2_slot, ctx_n1_slot, \
                  ctx_0_slot, ctx_p1_slot, ctx_p2_slot, predicate_slot, mark_slot, label_slot

    return reader


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    crf_cost, crf_dec = db_lstm(word_dict_len, label_dict_len, pred_len)

    #parameters = paddle.parameters.create([crf_cost, crf_dec])
    parameters = paddle.parameters.create(crf_cost)
    optimizer = paddle.optimizer.Momentum(momentum=0.01, learning_rate=2e-2)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)

        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=optimizer)

    reader_dict = {
        'word_data': 0,
        'ctx_n2_data': 1,
        'ctx_n1_data': 2,
        'ctx_0_data': 3,
        'ctx_p1_data': 4,
        'ctx_p2_data': 5,
        'verb_data': 6,
        'mark_data': 7,
        'target': 8,
    }
    #trn_reader = paddle.reader.batched(
    #    paddle.reader.shuffle(
    #        train_reader(), buf_size=8192), batch_size=2)
    trn_reader = paddle.reader.batched(train_reader(), batch_size=1)
    trainer.train(
        reader=trn_reader,
        cost=crf_cost,
        parameters=parameters,
        event_handler=event_handler,
        num_passes=10000,
        reader_dict=reader_dict)
    #cost=[crf_cost, crf_dec],


if __name__ == '__main__':
    main()
