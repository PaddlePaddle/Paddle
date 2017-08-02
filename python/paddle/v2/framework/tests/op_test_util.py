import paddle.v2.framework.core as core
import unittest
import numpy
import paddle.v2.framework.create_op_creation_methods as creation


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
            func = getattr(creation.op_creations, self.type, None)
            self.assertIsNotNone(func)

            scope = core.Scope()
            kwargs = dict()

            for in_name in func.all_input_args:
                if hasattr(self, in_name):
                    kwargs[in_name] = in_name
                    var = scope.new_var(in_name).get_tensor()
                    arr = getattr(self, in_name)
                    var.set_dims(arr.shape)
                    var.set(arr)
                else:
                    kwargs[in_name] = "@EMPTY@"

            for out_name in func.all_output_args:
                if hasattr(self, out_name):
                    kwargs[out_name] = out_name
                    scope.new_var(out_name).get_tensor()

            for attr_name in func.all_attr_args:
                if hasattr(self, attr_name):
                    kwargs[attr_name] = getattr(self, attr_name)

            op = func(**kwargs)

            op.infer_shape(scope)

            ctx = core.DeviceContext.cpu_context()
            op.run(scope, ctx)

            for out_name in func.all_output_args:
                actual = numpy.array(scope.find_var(out_name).get_tensor())
                expect = getattr(self, out_name)
                # TODO(qijun) The default decimal is 7, but numpy.dot and eigen.mul
                # has some diff, and could not pass unittest. So I set decimal 3 here.
                # And I will check this in future.
                numpy.testing.assert_almost_equal(actual, expect, decimal=3)

        obj.test_all = test_all
        return obj
