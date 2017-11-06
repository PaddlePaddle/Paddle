import paddle.v2.framework.layers as layers

__all__ = ["simple_img_conv_pool", "sequence_conv_pool"]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         pool_type='max',
                         main_program=None,
                         startup_program=None):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        act=act,
        main_program=main_program,
        startup_program=startup_program)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        main_program=main_program,
        startup_program=startup_program)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=None,
                   pool_stride=1,
                   pool_type=None,
                   main_program=None,
                   startup_program=None):
    """
    Image Convolution Group, Used for vgg net.
    """
    tmp = input
    assert isinstance(conv_num_filter, list) or \
        isinstance(conv_num_filter, tuple)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in xrange(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = layers.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            act=local_conv_act,
            main_program=main_program,
            startup_program=startup_program)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(
                input=tmp,
                act=conv_act,
                main_program=main_program,
                startup_program=startup_program)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(
                    x=tmp,
                    dropout_prob=drop_rate,
                    main_program=main_program,
                    startup_program=startup_program)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        main_program=main_program,
        startup_program=startup_program)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       act="sigmoid",
                       pool_type="max",
                       main_program=None,
                       startup_program=None):
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        act=act,
        main_program=main_program,
        startup_program=startup_program)

    pool_out = layers.sequence_pool(
        input=conv_out,
        pool_type=pool_type,
        main_program=main_program,
        startup_program=startup_program)
    return pool_out
