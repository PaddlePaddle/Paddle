from paddle.trainer_config_helpers import *

data = data_layer(name='data', size=3 * 14 * 14, height=14, width=14)

rois = data_layer(name='rois', size=10)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=3,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

roi_pool = roi_pool_layer(
    input=conv,
    rois=rois,
    pooled_width=7,
    pooled_height=7,
    spatial_scale=1. / 16)

outputs(roi_pool)
