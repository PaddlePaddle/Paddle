from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################

if not is_predict:
    data_dir = './data/'
    define_py_data_sources2(
        train_list=data_dir + 'train.list',
        test_list=data_dir + 'test.list',
        module='mnist_provider',
        obj='process')

######################Algorithm Configuration #############
# settings(
#    batch_size=128,
#    learning_rate=0.1 / 128.0,
#    learning_method=MomentumOptimizer(0.9),
#    regularization=L2Regularization(0.0005 * 128))
settings(
    batch_size=50,
    learning_rate=0.001,
    learning_method=AdamOptimizer())

#######################Network Configuration #############

data_size = 1 * 28 * 28
label_size = 10
img = data_layer(name='pixel', size=data_size)

# small_vgg is predined in trainer_config_helpers.network
# predict = small_vgg(input_image=img, num_channels=1, num_classes=label_size)

# light cnn
def light_cnn(input_image, num_channels, num_classes):
    def __light__(ipt, num_filter=128, times=1, conv_filter_size=3, dropouts=0, num_channels_=None):
        return img_conv_group(
            input=ipt,
            num_channels=num_channels_,
            pool_size=2,
            pool_stride=2,
            conv_padding=0,
            conv_num_filter=[num_filter] * times,
            conv_filter_size=conv_filter_size,
            conv_act=ReluActivation(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=MaxPooling())

    tmp = __light__(input_image, num_filter=128, num_channels_=num_channels)
    tmp = __light__(tmp, num_filter=128)
    tmp = __light__(tmp, num_filter=128)
    tmp = __light__(tmp, num_filter=128, conv_filter_size=1)

    #tmp = img_pool_layer(input=tmp, stride=2, pool_size=2, pool_type=MaxPooling())
    #tmp = dropout_layer(input=tmp, dropout_rate=0.5)
    tmp = fc_layer(input=tmp, size = num_classes, act=SoftmaxActivation())
    # tmp = fc_layer(input=tmp, size=512, layer_attr=ExtraAttr(drop_rate=0.5), act=LinearActivation())
    # tmp = batch_norm_layer(input=tmp, act=ReluActivation())
    # return fc_layer(input=tmp, size=num_classes, act=SoftmaxActivation())
    return tmp

predict = light_cnn(input_image=img, num_channels=1, num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    inputs(img, lbl)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)
