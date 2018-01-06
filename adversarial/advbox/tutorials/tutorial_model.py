################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""

A pure Paddlepaddle implementation of a neural network.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from advbox import Model

def main():
    """
	example main function
    """
    model_dir = "./mnist_model"
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_var_names, fetch_vars = fluid.io.load_inferfence_model(model_dir, exe)
    print(program)

if __name__ == "__main__":
    main()
