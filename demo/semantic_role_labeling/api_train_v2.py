import numpy
import paddle.v2 as paddle
from paddle.trainer_config_helpers.atts import ParamAttr

from mode_v2 import db_lstm

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

    label_dict_len = 500
    # define network topology
    output = db_lstm()
    target = paddle.layer.data(name='target', size=label_dict_len)
    crf_cost = paddle.layer.crf_layer(
        size=500,
        input=output,
        label=target,
        param_attr=paddle.attr.Param(
            name='crfw', initial_std=default_std, learning_rate=mix_hidden_lr))

    crf_dec = paddle.layer.crf_decoding_layer(
        name='crf_dec_l',
        size=label_dict_len,
        input=output,
        label=target,
        param_attr=paddle.attr.Param(name='crfw'))

    topo = [crf_cost, crf_dec]
    parameters = paddle.parameters.create(topo)
    optimizer = paddle.optimizer.Momentum(momentum=0.01, learning_rate=2e-2)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            para = parameters.get('___fc_2__.w0')
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost, para.mean())

        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=optimizer)

    trainer.train(
        train_data_reader=train_reader,
        batch_size=32,
        topology=topo,
        parameters=parameters,
        event_handler=event_handler,
        num_passes=10000,
        data_types=[],
        reader_dict={})


if __name__ == '__main__':
    main()
