#!/usr/bin/env python
from paddle.trainer_config_helpers import *

height = 224
width = 224
num_class = 1000
batch_size = get_config_arg('batch_size', int, 128)
use_gpu = get_config_arg('use_gpu', bool, True)
is_infer = get_config_arg("is_infer", bool, False)
num_samples = get_config_arg('num_samples', int, 2560)

args = {
    'height': height,
    'width': width,
    'color': True,
    'num_class': num_class,
    'is_infer': is_infer,
    'num_samples': num_samples
}
define_py_data_sources2(
    "train.list" if not is_infer else None,
    "test.list" if is_infer else None,
    module="provider",
    obj="process",
    args=args)

settings(
    batch_size=batch_size,
    learning_rate=0.01 / batch_size,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * batch_size))

conv_projection = conv_projection if use_gpu else img_conv_layer

def inception2(name, input, channels, \
    filter1,
    filter3R, filter3,
    filter5R, filter5,
    proj):

    conv1 = name + '_1'
    conv3r = name + '_3r'
    conv3 = name + '_3'
    conv5r = name + '_5r'
    conv5 = name + '_5'
    maxpool = name + '_max'
    convproj = name + '_proj'

    cov1 = img_conv_layer(
        name=conv1,
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter1,
        stride=1,
        padding=0)

    cov3r = img_conv_layer(
        name=conv3r,
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter3R,
        stride=1,
        padding=0)
    cov3 = img_conv_layer(
        name=conv3,
        input=cov3r,
        filter_size=3,
        num_filters=filter3,
        stride=1,
        padding=1)

    cov5r = img_conv_layer(
        name=conv5r,
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter5R,
        stride=1,
        padding=0)
    cov5 = img_conv_layer(
        name=conv5,
        input=cov5r,
        filter_size=5,
        num_filters=filter5,
        stride=1,
        padding=2)

    pool1 = img_pool_layer(
        name=maxpool,
        input=input,
        pool_size=3,
        num_channels=channels,
        stride=1,
        padding=1)
    covprj = img_conv_layer(
        name=convproj,
        input=pool1,
        filter_size=1,
        num_filters=proj,
        stride=1,
        padding=0)

    cat = concat_layer(name=name, input=[cov1, cov3, cov5, covprj])
    return cat

def inception(name, input, channels, \
    filter1,
    filter3R, filter3,
    filter5R, filter5,
    proj):

    cov1 = conv_projection(
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter1,
        stride=1,
        padding=0)

    cov3r = img_conv_layer(
        name=name + '_3r',
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter3R,
        stride=1,
        padding=0)
    cov3 = conv_projection(
        input=cov3r, filter_size=3, num_filters=filter3, stride=1, padding=1)

    cov5r = img_conv_layer(
        name=name + '_5r',
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter5R,
        stride=1,
        padding=0)
    cov5 = conv_projection(
        input=cov5r, filter_size=5, num_filters=filter5, stride=1, padding=2)

    pool1 = img_pool_layer(
        name=name + '_max',
        input=input,
        pool_size=3,
        num_channels=channels,
        stride=1,
        padding=1)
    covprj = conv_projection(
        input=pool1, filter_size=1, num_filters=proj, stride=1, padding=0)

    cat = concat_layer(
        name=name,
        input=[cov1, cov3, cov5, covprj],
        bias_attr=True if use_gpu else False,
        act=ReluActivation())
    return cat


data = data_layer(name="input", size=3 * height * width)

# stage 1
conv1 = img_conv_layer(
    name="conv1",
    input=data,
    filter_size=7,
    num_channels=3,
    num_filters=64,
    stride=2,
    padding=3)
pool1 = img_pool_layer(
    name="pool1", input=conv1, pool_size=3, num_channels=64, stride=2)

# stage 2
conv2_1 = img_conv_layer(
    name="conv2_1",
    input=pool1,
    filter_size=1,
    num_filters=64,
    stride=1,
    padding=0)
conv2_2 = img_conv_layer(
    name="conv2_2",
    input=conv2_1,
    filter_size=3,
    num_filters=192,
    stride=1,
    padding=1)
pool2 = img_pool_layer(
    name="pool2", input=conv2_2, pool_size=3, num_channels=192, stride=2)

# stage 3
ince3a = inception("ince3a", pool2, 192, 64, 96, 128, 16, 32, 32)
ince3b = inception("ince3b", ince3a, 256, 128, 128, 192, 32, 96, 64)
pool3 = img_pool_layer(
    name="pool3", input=ince3b, num_channels=480, pool_size=3, stride=2)

# stage 4
ince4a = inception("ince4a", pool3, 480, 192, 96, 208, 16, 48, 64)
ince4b = inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64, 64)
ince4c = inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64, 64)
ince4d = inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64, 64)
ince4e = inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128, 128)
pool4 = img_pool_layer(
    name="pool4", input=ince4e, num_channels=832, pool_size=3, stride=2)

# stage 5
ince5a = inception("ince5a", pool4, 832, 256, 160, 320, 32, 128, 128)
ince5b = inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128, 128)
pool5 = img_pool_layer(
    name="pool5",
    input=ince5b,
    num_channels=1024,
    pool_size=7,
    stride=7,
    pool_type=AvgPooling())

# We remove loss1 and loss2 for all system when testing benchmark
# output 1
# pool_o1 = img_pool_layer(name="pool_o1", input=ince4a, num_channels=512, pool_size=5, stride=3, pool_type=AvgPooling())
# conv_o1 = img_conv_layer(name="conv_o1", input=pool_o1, filter_size=1, num_filters=128, stride=1, padding=0)
# fc_o1 = fc_layer(name="fc_o1", input=conv_o1, size=1024, layer_attr=ExtraAttr(drop_rate=0.7), act=ReluActivation())
# out1 = fc_layer(name="output1", input=fc_o1,  size=1000, act=SoftmaxActivation())
# loss1 = cross_entropy(name='loss1', input=out1, label=lab, coeff=0.3) 

# output 2
#pool_o2 = img_pool_layer(name="pool_o2", input=ince4d, num_channels=528, pool_size=5, stride=3, pool_type=AvgPooling())
#conv_o2 = img_conv_layer(name="conv_o2", input=pool_o2, filter_size=1, num_filters=128, stride=1, padding=0)
#fc_o2 = fc_layer(name="fc_o2", input=conv_o2, size=1024, layer_attr=ExtraAttr(drop_rate=0.7), act=ReluActivation())
#out2 = fc_layer(name="output2", input=fc_o2, size=1000, act=SoftmaxActivation())
#loss2 = cross_entropy(name='loss2', input=out2, label=lab, coeff=0.3) 

# output 3
dropout = dropout_layer(name="dropout", input=pool5, dropout_rate=0.4)
out3 = fc_layer(
    name="output3", input=dropout, size=1000, act=SoftmaxActivation())

if is_infer:
    outputs(out3)
else:
    lab = data_layer(name="label", size=num_class)
    loss3 = cross_entropy(name='loss3', input=out3, label=lab)
    outputs(loss3)
