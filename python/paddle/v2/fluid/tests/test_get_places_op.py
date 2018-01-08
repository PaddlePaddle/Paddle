import paddle.v2.fluid as fluid
import decorators
import unittest


class TestGetPlaces(unittest.TestCase):
    @decorators.prog_scope()
    def test_get_places(self):
        places = fluid.layers.get_places()
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_main_program())
        self.assertEqual(places.type, fluid.core.VarDesc.VarType.PLACE_LIST)


if __name__ == '__main__':
    unittest.main()
