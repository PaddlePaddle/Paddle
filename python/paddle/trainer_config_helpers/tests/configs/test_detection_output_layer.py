from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

input_loc = data_layer(name='input_loc', size=16, height=16, width=1)

input_conf = data_layer(name='input_conf', size=8, height=1, width=8)

priorbox = data_layer(name='priorbox', size=32, height=4, width=8)

detout = detection_output_layer(
    input_loc=input_loc,
    input_conf=input_conf,
    priorbox=priorbox,
    num_classes=21,
    nms_threshold=0.45,
    nms_top_k=400,
    keep_top_k=200,
    confidence_threshold=0.01,
    background_id=0,
    name='test_detection_output')

outputs(detout)
