import paddle.v2.framework.layers as layers


def simple_img_conv_pool(input,
                         filter_size,
                         num_filters,
                         pool_size,
                         pool_stride,
                         act,
                         program=None):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        act=act,
        program=program)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type='max',
        pool_stride=pool_stride,
        program=program)
    return pool_out
