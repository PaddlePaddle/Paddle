from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

input_loc = data_layer(name='input_loc', size=21 * 4)

input_conf = data_layer(name='input_conf', size=21)

rois = data_layer(name='rois', size=10)

rcnn_loss = rcnn_loss_layer(
    rois=rois,
    input_loc=input_loc,
    input_conf=input_conf,
    loss_ratio=1,
    num_classes=21,
    background_id=0,
    name='test_rcnn_loss')

outputs(rcnn_loss)
