import layers

__all__ = [
    "simple_img_conv_pool",
    "sequence_conv_pool",
]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         param_attr=None,
                         pool_type='max',
                         use_cudnn=False):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act,
        use_cudnn=use_cudnn)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   param_attr=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=None,
                   conv_use_cudnn=False,
                   pool_stride=1,
                   pool_type=None,
                   pool_use_cudnn=False):
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
    param_attr = __extend_list__(param_attr)
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
            param_attr=param_attr[i],
            act=local_conv_act,
            use_cudnn=conv_use_cudnn)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=pool_use_cudnn)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       param_attr=None,
                       act="sigmoid",
                       pool_type="max"):
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act)

    pool_out = layers.sequence_pool(input=conv_out, pool_type=pool_type)
    return pool_out
