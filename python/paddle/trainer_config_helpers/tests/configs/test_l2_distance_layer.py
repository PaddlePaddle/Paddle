from paddle.trainer_config_helpers import *

outputs(
    l2_distance_layer(
        x=data_layer(
            name='x', size=128), y=data_layer(
                name='y', size=128)))
