import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.io import save_inference_model, load_inference_model
import paddle.v2.framework.executor as executor
import unittest
import numpy as np

x_data = np.array([[1, 1], [1, 2], [3, 4], [5, 2]]).astype("float32")
y_data = np.array([[-2], [-3], [-7], [-7]]).astype("float32")

place = core.CPUPlace()
tensor_x = core.LoDTensor()
tensor_x.set(x_data, place)
# print tensor_x.get_dims()

tensor_y = core.LoDTensor()
tensor_y.set(y_data, place)
# print tensor_y.get_dims()

exe = executor.Executor(place)
[infer_prog, feed_var_names, fetch_vars] = load_inference_model(
    "./fit_line_infer_model", exe)

outs = exe.run(infer_prog,
               feed={feed_var_names[0]: tensor_x,
                     feed_var_names[1]: tensor_y},
               fetch_list=fetch_vars)
out = np.array(outs[0])
print ""
print out
