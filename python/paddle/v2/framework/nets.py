import paddle.v2.framework.layers as layers


def simple_img_conv_pool(input,
                         filter_size,
                         num_filters,
                         pool_size,
                         pool_stride,
                         act,
                         program=None):
    pre_pool = layers.conv2d_layer(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        program=program,
        act=act)
    res = layers.pool2d(
        x=pre_pool, pooling_type='max', ksize=pool_size, strides=pool_stride)
    return res
