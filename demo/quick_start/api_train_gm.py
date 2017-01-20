import random
import cPickle
import os
import paddle.v2 as paddle


class FileReader(object):
    """
    :type word_dict: dict
    :type __pool__: list
    """

    def __init__(self, word_dict, filename, batch_size, should_shuffle=True):
        if isinstance(word_dict, basestring):
            self.word_dict = FileReader.read_from_dict(word_dict)
        else:
            self.word_dict = word_dict
        self.__should_shuffle__ = should_shuffle
        self.__batch_size__ = batch_size

        self.__pool__ = self.load_all_data(filename)
        self.__idx__ = 0

    def load_all_data(self, filename):
        def __mapper__(line):
            label, sentence = line.split('\t')
            label = int(label)
            word_ids = filter(lambda x: x is not None,
                              map(lambda x: self.word_dict.get(x, None),
                                  sentence.split()))
            return word_ids, label

        if filename[-3:] == 'txt':
            with open(filename, 'r') as f:
                ret_val = map(__mapper__, f)
            with open("%s.pkl" % filename[:-4], 'wb') as f:
                cPickle.dump(ret_val, f, cPickle.HIGHEST_PROTOCOL)
            return ret_val
        elif filename[-3:] == 'pkl':
            with open(filename, 'rb') as f:
                return cPickle.load(f)

    def __iter__(self):
        self.reset()
        return self

    def reset(self):
        if self.__should_shuffle__:
            random.shuffle(self.__pool__)
        self.__idx__ = 0

    def next(self):
        if self.__idx__ < len(self.__pool__):
            end = min(self.__idx__ + self.__batch_size__, len(self.__pool__))
            start = self.__idx__
            self.__idx__ = end
            return self.__pool__[start:end]
        else:
            raise StopIteration()

    @staticmethod
    def read_from_dict(fn):
        if os.path.exists(fn + '.pkl'):
            with open(fn + '.pkl', 'rb') as f:
                return cPickle.load(f)
        else:
            ret_val = dict()
            with open(fn, 'r') as f:
                for i, line in enumerate(f):
                    w = line.split()[0]
                    ret_val[w] = i
            with open(fn + '.pkl', 'wb') as f:
                cPickle.dump(ret_val, f, cPickle.HIGHEST_PROTOCOL)
            return ret_val


def optimizer_config():
    paddle.config.settings(
        batch_size=1,
        learning_rate=1e-4,
        learning_method=paddle.config.RMSPropOptimizer())


def bow_config(dict_size):
    def __impl__():
        sentence = paddle.config.data_layer(name='sentence', size=dict_size)
        inference = paddle.config.fc_layer(
            input=sentence,
            size=2,
            act=paddle.config.SoftmaxActivation(),
            param_attr=paddle.config.ParamAttr(sparse_update=True))
        cost = paddle.config.classification_cost(
            input=inference,
            label=paddle.config.data_layer(
                name='label', size=2))
        paddle.config.outputs(cost)

    return __impl__


def swap_batch(batch):
    for each_item in batch:
        a, b = each_item
        yield b, a


def main():
    print 'Loading data into memory'
    train_file_name = './data/train.pkl' if os.path.exists(
        './data/train.pkl') else './data/train.txt'

    test_file_name = './data/test.pkl' if os.path.exists(
        './data/test.pkl') else './data/test.txt'

    train_reader = FileReader(
        "./data/dict.txt", filename=train_file_name, batch_size=1024)
    test_reader = FileReader(
        train_reader.word_dict, filename=test_file_name, batch_size=1024)

    print 'Done.'

    paddle.raw.initPaddle('--use_gpu=0', '--trainer_count=3')

    optimizer_proto = paddle.config.parse_optimizer(
        optimizer_conf=optimizer_config)
    optimizer_conf = paddle.raw.OptimizationConfig.createFromProto(
        optimizer_proto)
    __tmp_optimizer__ = paddle.raw.ParameterOptimizer.create(optimizer_conf)
    assert isinstance(__tmp_optimizer__, paddle.raw.ParameterOptimizer)
    enable_types = __tmp_optimizer__.getParameterTypes()

    model_proto = paddle.config.parse_network(
        network_conf=bow_config(len(train_reader.word_dict)))

    for param in model_proto.parameters:
        if param.sparse_remote_update:
            # disable sparse remote update, when local
            param.sparse_remote_update = False

    gradient_machine = paddle.raw.GradientMachine.createFromConfigProto(
        model_proto, paddle.raw.CREATE_MODE_NORMAL, enable_types)
    assert isinstance(gradient_machine, paddle.raw.GradientMachine)
    gradient_machine.randParameters()

    updater = paddle.raw.ParameterUpdater.createLocalUpdater(optimizer_conf)
    assert isinstance(updater, paddle.raw.ParameterUpdater)

    input_order = model_proto.input_layer_names
    input_types = {
        'sentence':
        paddle.data.sparse_binary_vector(len(train_reader.word_dict)),
        'label': paddle.data.integer_value(2)
    }

    tmp = []
    for each in input_order:
        tmp.append(input_types[each])

    input_types = tmp

    converter = paddle.data.DataProviderConverter(input_types=input_types)

    input_order_for_data = ['sentence', 'label']
    switcher = None
    if input_order_for_data != input_order:
        switcher = swap_batch

    updater.init(gradient_machine)

    gradient_machine.start()

    train_evaluator = gradient_machine.makeEvaluator()
    test_evaluator = gradient_machine.makeEvaluator()
    assert isinstance(train_evaluator, paddle.raw.Evaluator)
    assert isinstance(test_evaluator, paddle.raw.Evaluator)

    train_evaluate_period = 100

    out_args = paddle.raw.Arguments.createArguments(0)
    assert isinstance(out_args, paddle.raw.Arguments)
    for pass_id in xrange(10):
        updater.startPass()
        for batch_id, data_batch in enumerate(train_reader):
            if switcher is not None:
                data_batch = switcher(data_batch)

            updater.startBatch(len(data_batch))

            in_args = converter(data_batch)

            if batch_id % train_evaluate_period == 0:
                train_evaluator.start()

            gradient_machine.forwardBackward(in_args, out_args,
                                             paddle.raw.PASS_TRAIN)

            gradient_machine.eval(train_evaluator)

            cost = out_args.sumCosts() / len(data_batch)

            if batch_id % train_evaluate_period == 0:
                print 'Pass=%d Batch=%d Cost=%f' % (pass_id, batch_id,
                                                    cost), train_evaluator
                train_evaluator.finish()

            gradient_machine.eval(train_evaluator)

            for each_param in gradient_machine.getParameters():
                updater.update(each_param)

            updater.finishBatch(cost)

        print 'Pass=%d Batch=%d Cost=%f' % (pass_id, batch_id,
                                            cost), train_evaluator
        updater.catchUpWith()

        test_evaluator.start()
        for data_batch in test_reader:
            if switcher is not None:
                data_batch = switcher(data_batch)

            in_args = converter(data_batch)
            gradient_machine.forward(in_args, out_args, paddle.raw.PASS_TEST)
            gradient_machine.eval(test_evaluator)

        print 'Test Pass=%d' % pass_id, test_evaluator

        print 'Saving parameters.'
        for param in gradient_machine.getParameters():
            assert isinstance(param, paddle.raw.Parameter)
            save_name = "%d_%s" % (pass_id, param.getName())
            param.save(save_name)
        print 'Done.'

        test_evaluator.finish()

        updater.finishPass()
    gradient_machine.finish()


if __name__ == '__main__':
    main()
