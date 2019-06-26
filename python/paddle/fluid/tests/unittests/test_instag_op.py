
from __future__ import print_function
 
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
 

class TestInstagOp(OpTest):
	def setUp(self):
		self.op_type = 'instag'
		batch_size = 4
		x1_embed_size = 4
		fc_cnt = 2
		x1 = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]).astype('double')

		x2 = fluid.create_lod_tensor(np.array([[1],[1],[2],[2]]).astype('int64'), [[1,1,1,1]], fluid.CPUPlace())

		x3 = fluid.create_lod_tensor(np.array([[1],[2]]).astype('int64'), [[1,1]], fluid.CPUPlace())


		out = np.array([[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]], [[0,0,0,0], [0,0,0,0],[1,1,1,1],[1,1,1,1]]]).astype('double')

		self.inputs = {
			'X1': x1,
			'X2': x2,
			'X3': x3,
		}			
		
		self.outputs = {
			'Out' : out,
		}		

	def test_check_output(self):
		self.check_output()


if __name__ == '__main__':
	unittest.main()
