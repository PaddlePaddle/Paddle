from paddle.trainer_config_helpers import *

data = data_layer(name='data', size=3 * 14 * 14, height=14, width=14)

rois = data_layer(name='rois', size=10)

roi_pool = roi_pool_layer(
    input=data,
    rois=rois,
    pooled_width=7,
    pooled_height=7,
    spatial_scale=1. / 16)

outputs(roi_pool)
