from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-4)

#data = data_layer(name='data', size=180, width=30, height=6)
#batchNorm = batch_norm_layer(data, num_channels=1)
#outputs(batchNorm)

data3D = data_layer(name='data3D', size=120 * 3, width=20, height=6, depth=3)
batchNorm3D = batch_norm_layer(data3D, num_channels=1, img3D=True)
outputs(batchNorm3D)
