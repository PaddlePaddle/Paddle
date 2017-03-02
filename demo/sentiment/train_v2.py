import sys
from os.path import join as join_path
import paddle.trainer_config_helpers.attrs as attrs
from paddle.trainer_config_helpers.poolings import MaxPooling
import paddle.v2 as paddle
import paddle.v2.layer as layer
import paddle.v2.activation as activation
import paddle.v2.data_type as data_type


def sequence_conv_pool(input,
                       input_size,
                       context_len,
                       hidden_size,
                       name=None,
                       context_start=None,
                       pool_type=None,
                       context_proj_layer_name=None,
                       context_proj_param_attr=False,
                       fc_layer_name=None,
                       fc_param_attr=None,
                       fc_bias_attr=None,
                       fc_act=None,
                       pool_bias_attr=None,
                       fc_attr=None,
                       context_attr=None,
                       pool_attr=None):
    """
    Text convolution pooling layers helper.

    Text input => Context Projection => FC Layer => Pooling => Output.

    :param name: name of output layer(pooling layer name)
    :type name: basestring
    :param input: name of input layer
    :type input: LayerOutput
    :param context_len: context projection length. See
                        context_projection's document.
    :type context_len: int
    :param hidden_size: FC Layer size.
    :type hidden_size: int
    :param context_start: context projection length. See
                          context_projection's context_start.
    :type context_start: int or None
    :param pool_type: pooling layer type. See pooling_layer's document.
    :type pool_type: BasePoolingType.
    :param context_proj_layer_name: context projection layer name.
                                    None if user don't care.
    :type context_proj_layer_name: basestring
    :param context_proj_param_attr: context projection parameter attribute.
                                    None if user don't care.
    :type context_proj_param_attr: ParameterAttribute or None.
    :param fc_layer_name: fc layer name. None if user don't care.
    :type fc_layer_name: basestring
    :param fc_param_attr: fc layer parameter attribute. None if user don't care.
    :type fc_param_attr: ParameterAttribute or None
    :param fc_bias_attr: fc bias parameter attribute. False if no bias,
                         None if user don't care.
    :type fc_bias_attr: ParameterAttribute or None
    :param fc_act: fc layer activation type. None means tanh
    :type fc_act: BaseActivation
    :param pool_bias_attr: pooling layer bias attr. None if don't care.
                           False if no bias.
    :type pool_bias_attr: ParameterAttribute or None.
    :param fc_attr: fc layer extra attribute.
    :type fc_attr: ExtraLayerAttribute
    :param context_attr: context projection layer extra attribute.
    :type context_attr: ExtraLayerAttribute
    :param pool_attr: pooling layer extra attribute.
    :type pool_attr: ExtraLayerAttribute
    :return: output layer name.
    :rtype: LayerOutput
    """
    # Set Default Value to param
    context_proj_layer_name = "%s_conv_proj" % name \
        if context_proj_layer_name is None else context_proj_layer_name

    with layer.mixed(
            name=context_proj_layer_name,
            size=input_size * context_len,
            act=activation.Linear(),
            layer_attr=context_attr) as m:
        m += layer.context_projection(
            input=input,
            context_len=context_len,
            context_start=context_start,
            padding_attr=context_proj_param_attr)

    fc_layer_name = "%s_conv_fc" % name \
        if fc_layer_name is None else fc_layer_name
    fl = layer.fc(name=fc_layer_name,
                  input=m,
                  size=hidden_size,
                  act=fc_act,
                  layer_attr=fc_attr,
                  param_attr=fc_param_attr,
                  bias_attr=fc_bias_attr)

    return layer.pooling(
        name=name,
        input=fl,
        pooling_type=pool_type,
        bias_attr=pool_bias_attr,
        layer_attr=pool_attr)


def convolution_net(input_dim,
                    class_dim=2,
                    emb_dim=128,
                    hid_dim=128,
                    is_predict=False):
    data = layer.data("word", data_type.integer_value_sequence(input_dim))
    emb = layer.embedding(input=data, size=emb_dim)
    conv_3 = sequence_conv_pool(
        input=emb, input_size=emb_dim, context_len=3, hidden_size=hid_dim)
    conv_4 = sequence_conv_pool(
        input=emb, input_size=emb_dim, context_len=4, hidden_size=hid_dim)
    output = layer.fc(input=[conv_3, conv_4],
                      size=class_dim,
                      act=activation.Softmax())
    lbl = layer.data("label", data_type.integer_value(2))
    cost = layer.classification_cost(input=output, label=lbl)
    return cost


def stacked_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3,
                     is_predict=False):
    """
    A Wrapper for sentiment classification task.
    This network uses bi-directional recurrent network,
    consisting three LSTM layers. This configure is referred to
    the paper as following url, but use fewer layrs.
        http://www.aclweb.org/anthology/P15-1109

    input_dim: here is word dictionary dimension.
    class_dim: number of categories.
    emb_dim: dimension of word embedding.
    hid_dim: dimension of hidden layer.
    stacked_num: number of stacked lstm-hidden layer.
    is_predict: is predicting or not.
                Some layers is not needed in network when predicting.
    """
    assert stacked_num % 2 == 1

    layer_attr = attrs.ExtraLayerAttribute(drop_rate=0.5)
    fc_para_attr = attrs.ParameterAttribute(learning_rate=1e-3)
    lstm_para_attr = attrs.ParameterAttribute(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = attrs.ParameterAttribute(initial_std=0., l2_rate=0.)
    relu = activation.Relu()
    linear = activation.Linear()

    data = layer.data("word", data_type.integer_value_sequence(input_dim))
    emb = layer.embedding(input=data, size=emb_dim)

    fc1 = layer.fc(input=emb, size=hid_dim, act=linear, bias_attr=bias_attr)
    lstm1 = layer.lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr, layer_attr=layer_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = layer.fc(input=inputs,
                      size=hid_dim,
                      act=linear,
                      param_attr=para_attr,
                      bias_attr=bias_attr)
        lstm = layer.lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr,
            layer_attr=layer_attr)
        inputs = [fc, lstm]

    fc_last = layer.pooling(input=inputs[0], pooling_type=MaxPooling())
    lstm_last = layer.pooling(input=inputs[1], pooling_type=MaxPooling())
    output = layer.fc(input=[fc_last, lstm_last],
                      size=class_dim,
                      act=activation.Softmax(),
                      bias_attr=bias_attr,
                      param_attr=para_attr)

    lbl = layer.data("label", data_type.integer_value(2))
    cost = layer.classification_cost(input=output, label=lbl)
    return cost


def data_reader(data_file, dict_file):
    def reader():
        with open(dict_file, 'r') as fdict, open(data_file, 'r') as fdata:
            dictionary = dict()
            for i, line in enumerate(fdict):
                dictionary[line.split('\t')[0]] = i

            for line_count, line in enumerate(fdata):
                label, comment = line.strip().split('\t\t')
                label = int(label)
                words = comment.split()
                word_slot = [dictionary[w] for w in words if w in dictionary]
                yield (word_slot, label)

    return reader


if __name__ == '__main__':
    # data file
    train_file = "./data/pre-imdb/train_part_000"
    test_file = "./data/pre-imdb/test_part_000"
    dict_file = "./data/pre-imdb/dict.txt"
    labels = "./data/pre-imdb/labels.list"

    # init
    paddle.init(use_gpu=True, trainer_count=4)

    # network config
    dict_dim = len(open(dict_file).readlines())
    class_dim = len(open(labels).readlines())

    # Please choose the way to build the network
    # by uncommenting the corresponding line.
    cost = convolution_net(dict_dim, class_dim=class_dim)
    # cost = stacked_lstm_net(dict_dim, class_dim=class_dim, stacked_num=3)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.reader.batched(
                    data_reader(test_file, dict_file), batch_size=128),
                reader_dict={'word': 0,
                             'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                data_reader(train_file, dict_file), buf_size=4096),
            batch_size=128),
        event_handler=event_handler,
        reader_dict={'word': 0,
                     'label': 1},
        num_passes=10)
