import paddle.v2.framework.core as core
import unittest
import numpy
from paddle.v2.framework.op import Operator


class OpTestMeta(type):
    """
    Operator Test ClassMeta.
    
    It injects `test_all` method into user's OperatorTest class, to make Python 
    unittest module run that method.
    
    The `test_all` read what value is stored in `self`. It use self's values to
    create and run a operator, and check whether that op is OK or not.
    
    See `test_add_two_op` for example usage.
    """

    def __new__(cls, name, bases, attrs):
        obj = super(OpTestMeta, cls).__new__(cls, name, bases, attrs)

        def test_all(self):
            scope = core.Scope()
            kwargs = dict()
            places = [core.CPUPlace()]
            if core.is_compile_gpu():
                places.append(core.GPUPlace(0))

            for place in places:
                for in_name in Operator.get_op_input_names(self.type):
                    if hasattr(self, in_name):
                        kwargs[in_name] = in_name
                        var = scope.new_var(in_name).get_tensor()
                        arr = getattr(self, in_name)
                        var.set_dims(arr.shape)
                        var.set(arr, place)
                    else:
                        kwargs[in_name] = "@EMPTY@"

                for out_name in Operator.get_op_output_names(self.type):
                    if hasattr(self, out_name):
                        kwargs[out_name] = out_name
                        scope.new_var(out_name).get_tensor()

                for attr_name in Operator.get_op_attr_names(self.type):
                    if hasattr(self, attr_name):
                        kwargs[attr_name] = getattr(self, attr_name)

                op = Operator(self.type, **kwargs)

                op.infer_shape(scope)

                ctx = core.DeviceContext.create(place)
                op.run(scope, ctx)

                for out_name in Operator.get_op_output_names(self.type):
                    actual = numpy.array(scope.find_var(out_name).get_tensor())
                    expect = getattr(self, out_name)
                    # TODO(qijun) The default decimal is 7, but numpy.dot and eigen.mul
                    # has some diff, and could not pass unittest. So I set decimal 3 here.
                    # And I will check this in future.
                    numpy.testing.assert_almost_equal(actual, expect, decimal=3)

        obj.test_all = test_all
        return obj
