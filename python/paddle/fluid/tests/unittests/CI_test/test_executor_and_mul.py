'''
        self.assertEqual((100, 100), res.shape)
        self.assertTrue(numpy.allclose(res, numpy.dot(a_np, b_np)))
        self.assertTrue(numpy.allclose(res_array[0], a_np))
        self.assertTrue(numpy.allclose(res_array[1], b_np))
        self.assertTrue(numpy.allclose(res_array[2], res))
'''