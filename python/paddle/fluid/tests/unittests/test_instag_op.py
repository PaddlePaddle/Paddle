
from __future__ import print_function
 
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import gradient_checker 
from decorator_helper import prog_scope

class TestInstagOp(OpTest):
	def setUp(self):
		self.op_type = 'instag'
		batch_size = 4
		x1_embed_size = 4
		fc_cnt = 2
		x1 = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]).astype('double')

		x2 = np.array([[1, 2], [1, -1], [2, -1], [2, -1]]).astype('int64')
		
		x3 = np.array([[1, -1], [1, 2]]).astype('int64')

		out = np.array([[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]], [[1,1,1,1], [1,1,1,1],[1,1,1,1],[1,1,1,1]]]).astype('double')

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

	def test_check_grad(self):
        	self.check_grad(['X1'], 'Out', no_grad_set=set(['X2', 'X3']))

if __name__ == '__main__':
		unittest.main()
