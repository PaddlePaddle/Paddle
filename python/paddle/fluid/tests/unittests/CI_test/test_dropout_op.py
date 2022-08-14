'''
            self.assertTrue(np.allclose(x_out, x_in_np))


                self.assertTrue(np.allclose(fetches[0], res_np))
            self.assertTrue(np.allclose(fetches2[0], res_np2))


                self.assertTrue(np.allclose(res.numpy(), res_np))
            self.assertTrue(np.allclose(res10.numpy(), res_np2))


                self.assertTrue(np.allclose(result.numpy(), result_np))


                self.assertTrue(np.allclose(fetches[0], res_np))


                self.assertTrue(np.allclose(res.numpy(), res_np))


                self.assertTrue(np.allclose(result.numpy(), result_np))


                self.assertTrue(np.allclose(fetches[0], res_np))


                self.assertTrue(np.allclose(res.numpy(), res_np))


                self.assertTrue(np.allclose(result.numpy(), result_np))


                self.assertTrue(np.allclose(fetches[0], res_np))


            self.assertTrue(np.allclose(fetches[0], res_np3))


                self.assertTrue(np.allclose(res.numpy(), res_np))
            self.assertTrue(np.allclose(res3.numpy(), res_np3))


                self.assertTrue(np.allclose(result.numpy(), result_np))



                self.assertTrue(np.allclose(out1, out2))



                self.assertTrue(
                    np.allclose(input.gradient(),
                                self.cal_grad_upscale_train(mask.numpy(),
                                                            prob)))


                    self.assertTrue(
                        np.allclose(
                            input.gradient(),
                            self.cal_grad_upscale_train(mask.numpy(), prob)))


                self.assertTrue(
                    np.allclose(input.gradient(),
                                self.cal_grad_upscale_train(mask.numpy(),
                                                            prob)))


                    self.assertTrue(
                        np.allclose(
                            input.gradient(),
                            self.cal_grad_upscale_train(mask.numpy(), prob)))


            self.assertTrue(np.array_equal(static_res, dygraph_res))


        self.assertEqual(np.sum(index0), 390094540)
        self.assertEqual(np.sum(index1), 12871475125)
        self.assertEqual(np.sum(index2), 12872777397)
        self.assertEqual(np.sum(out), 16778744.0)


        self.assertTrue(np.allclose(out[10, 100, 500:510], expect))


        self.assertEqual(np.sum(index0), 260065137)
        self.assertEqual(np.sum(index1), 8582636095)
        self.assertEqual(np.sum(index2), 8582219962)
        self.assertEqual(np.sum(out), 16778396.563660286)


        self.assertTrue(np.allclose(out[20, 100, 500:510], expect))


        self.assertEqual(np.sum(index0), 130086900)
        self.assertEqual(np.sum(index1), 4291190105)
        self.assertEqual(np.sum(index2), 4292243807)


        self.assertTrue(np.allclose(out[0, 100, 500:510], expect))
'''