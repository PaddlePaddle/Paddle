import op_test
import unittest
import numpy


def create_test_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            a = numpy.random.random(size=(10, 7)).astype(typename)
            b = numpy.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}".format(op_type, typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {'float32', 'float64', 'int32', 'int64'}:
    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b)
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b)

if __name__ == '__main__':
    unittest.main()
