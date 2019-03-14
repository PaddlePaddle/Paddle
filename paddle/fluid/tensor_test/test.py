
import os
import sys

third_lib_path = '/Users/peizhilin/Desktop/me/mac_build/paddle/fluid/tensor_test/'
os.environ['PATH'] += ':' + third_lib_path
sys.path.append(third_lib_path)

import tensor_test

t = tensor_test.Tensor()
print t
