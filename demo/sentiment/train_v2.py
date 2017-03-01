from os.path import join as join_path
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
    lbl = layer.data("label", data_type.integer_value(1))
    cost = layer.classification_cost(input=output, label=lbl)
    return cost


def data_reader():
    data_dir = "./data/pre-imdb"
    train_file = "train_part_000"
    test_file = "test_part_000"
    dict_file = "dict.txt"
    train_file = join_path(data_dir, train_file)
    test_file = join_path(data_dir, test_file)
    dict_file = join_path(data_dir, dict_file)

    with open(dict_file, 'r') as fdict, open(train_file, 'r') as fdata:
        dictionary = dict()
        for i, line in enumerate(fdict):
            dictionary[line.split('\t')[0]] = i

        for line_count, line in enumerate(fdata):
            label, comment = line.strip().split('\t\t')
            label = int(label)
            words = comment.split()
            word_slot = [dictionary[w] for w in words if w in dictionary]
            yield (word_slot, label)


if __name__ == '__main__':
    data_dir = "./data/pre-imdb"
    train_list = "train.list"
    test_list = "test.list"
    dict_file = "dict.txt"
    dict_dim = len(open(join_path(data_dir, "dict.txt")).readlines())
    class_dim = len(open(join_path(data_dir, 'labels.list')).readlines())
    is_predict = False

    # init
    paddle.init(use_gpu=True, trainer_count=4)

    # network config
    cost = convolution_net(dict_dim, class_dim=class_dim, is_predict=is_predict)

    # create parameters
    parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                data_reader, buf_size=4096), batch_size=128),
        event_handler=event_handler,
        reader_dict={'word': 0,
                     'label': 1},
        num_passes=10)
