from paddle.utils.merge_model import merge_v2_model

from mnist_v2 import network

net = network(is_infer=True)
param_file = "models/params_pass_4.tar"
output_file = "output.paddle.model"
merge_v2_model(net, param_file, output_file)
