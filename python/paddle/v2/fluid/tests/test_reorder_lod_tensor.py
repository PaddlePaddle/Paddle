import unittest
import paddle.v2.fluid as fluid
import numpy


class TestReorderLoDTensor(unittest.TestCase):
    def test_reorder(self):
        dat = fluid.layers.data(name='input', shape=[1], lod_level=2)
        dat.stop_gradient = False
        rank_dat = fluid.layers.data(name='ref', shape=[1], lod_level=1)
        table = fluid.layers.lod_rank_table(rank_dat)
        new_dat = fluid.layers.reorder_lod_tensor_by_rank(
            x=dat, rank_table=table)
        loss = fluid.layers.mean(x=new_dat)
        fluid.backward.append_backward_ops(loss=loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_startup_program())

        ref = fluid.Tensor()
        ref_lod = [0, 3, 4, 7, 8, 14]
        ref.set_lod([ref_lod])

        ref.set(numpy.random.random(size=[14, 1]).astype('float32'), cpu)
        input = fluid.Tensor()
        lod_level_0 = numpy.random.randint(low=1, high=5, size=5)
        lod_level_0 = [0] + numpy.cumsum(lod_level_0).tolist()
        lod_level_1 = numpy.random.randint(low=1, high=5, size=lod_level_0[-1])
        lod_level_1 = [0] + numpy.cumsum(lod_level_1).tolist()

        input.set_lod([lod_level_0, lod_level_1])
        input.set(
            numpy.random.random(size=[lod_level_1[-1], 1]).astype('float32'),
            cpu)

        ig = exe.run(fluid.default_main_program(),
                     feed={'input': input,
                           'ref': ref},
                     fetch_list=['input@GRAD'],
                     return_numpy=False)[0]
        self.assertAlmostEqual(numpy.array(ig).sum(), 1.0, delta=0.001)
        self.assertEqual(input.lod(), ig.lod())


if __name__ == '__main__':
    unittest.main()
