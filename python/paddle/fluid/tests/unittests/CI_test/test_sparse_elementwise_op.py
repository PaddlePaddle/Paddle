'''
            self.assertTrue(
                np.allclose(expect_res.numpy(),
                            actual_res.to_dense().numpy(),
                            equal_nan=True))


                self.assertTrue(
                    np.allclose(dense_x.grad.numpy(),
                                csr_x.grad.to_dense().numpy(),
                                equal_nan=True))
                self.assertTrue(
                    np.allclose(dense_y.grad.numpy(),
                                csr_y.grad.to_dense().numpy(),
                                equal_nan=True))


                self.assertTrue(
                    np.allclose(expect_res.numpy(),
                                actual_res.to_dense().numpy(),
                                equal_nan=True))
                self.assertTrue(
                    np.allclose(dense_x.grad.numpy(),
                                coo_x.grad.to_dense().numpy(),
                                equal_nan=True))
                self.assertTrue(
                    np.allclose(dense_y.grad.numpy(),
                                coo_y.grad.to_dense().numpy(),
                                equal_nan=True))
'''