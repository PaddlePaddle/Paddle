import paddle.v2.framework.layers as layers

__all__ = ["simple_img_conv_pool", "sequence_conv_pool"]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         program=None,
                         init_program=None):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        act=act,
        program=program,
        init_program=init_program)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type='max',
        pool_stride=pool_stride,
        program=program,
        init_program=init_program)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       pool_type,
                       pool_size,
                       pool_stride,
                       act,
                       program=None,
                       init_program=None):
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        act=act,
        program=program,
        init_program=init_program)

    pool_out = layers.sequence_pool(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        program=program,
        init_program=init_program)
    return pool_out
