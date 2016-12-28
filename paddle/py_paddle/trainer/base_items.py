"""
Some basically items.
"""
from .base import RunnerItem
from .. import swig_paddle as api

__all__ = ['init_runner_item', 'ContextWrapper', 'BaseRunnerItem']


def init_runner_item():
    def __impl__(func):
        class __ImplItem__(RunnerItem):
            __doc__ = func.__doc__

            def __init__(self, **kwargs):
                RunnerItem.__init__(self)
                self.__kwargs__ = kwargs

            def initialize(self, context, next_callback):
                func(context=context, **self.__kwargs__)
                next_callback(context=context)

        return __ImplItem__

    return __impl__


class ContextWrapper(object):
    """
    Strong typed wrapper to read/write context value.

    @TODO(yuyang18): Use Cython to implement this class, make it directly access
                     a C struct.
    """

    def __init__(self, context):
        self.real_context = context

    def gradient_machine(self, field_name='gradient_machine'):
        """
        Get Gradient Machine
        :param field_name:
        :return:
        :rtype: api.GradientMachine
        """
        return self.get_field_with_type(
            field_name=field_name, tp=api.GradientMachine)

    def set_gradient_machine(self, machine, field_name='gradient_machine'):
        self.set_field_with_type(
            field_name=field_name, value=machine, tp=api.GradientMachine)

    def updater(self, field_name='updater'):
        """
        Get Parameter Updater
        :param field_name:
        :return:
        :rtype: api.ParameterUpdater
        """
        return self.get_field_with_type(
            field_name=field_name, tp=api.ParameterUpdater)

    def set_updater(self, updater, field_name='updater'):
        self.set_field_with_type(
            field_name=field_name, value=updater, tp=api.ParameterUpdater)

    def set_field_with_type(self, field_name, value, tp, must_not_set=True):
        assert not must_not_set or not hasattr(self.real_context, field_name)
        assert isinstance(value, tp)
        setattr(self.real_context, field_name, value)

    def set_updater_callback(self,
                             updater_callback,
                             field_name='updater_callback'):
        assert callable(updater_callback)
        setattr(self.real_context, field_name, updater_callback)

    def updater_callback(self, field_name='updater_callback'):
        """

        :param field_name:
        :return:
        :rtype: callable
        """
        cb = getattr(self.real_context, field_name, None)
        assert callable(cb)
        return cb

    def batch_size(self, field_name='current_batch_size'):
        """

        :param field_name:
        :return:
        :rtype: int
        """
        return self.get_field_with_type(field_name=field_name, tp=int)

    def set_batch_size(self, batch_size, field_name='current_batch_size'):
        self.set_field_with_type(
            field_name=field_name, value=batch_size, tp=int, must_not_set=False)

    def cost(self, field_name='current_cost'):
        """

        :param field_name:
        :return:
        :rtype: float
        """
        return self.get_field_with_type(field_name=field_name, tp=float)

    def set_cost(self, cost, field_name='current_cost'):
        self.set_field_with_type(
            field_name=field_name, value=cost, tp=float, must_not_set=False)

    def reset_batch_id(self, field_name='current_batch_id'):
        self.set_field_with_type(
            field_name=field_name, value=0, tp=int, must_not_set=False)

    def batch_id(self, field_name='current_batch_id'):
        return self.get_field_with_type(field_name=field_name, tp=int)

    def increase_batch_id(self, field_name='current_batch_id'):
        self.set_field_with_type(
            field_name=field_name,
            value=self.batch_id(field_name=field_name) + 1,
            tp=int,
            must_not_set=False)

    def reset_pass_id(self, field_name='current_pass_id'):
        self.set_field_with_type(
            field_name=field_name, value=0, tp=int, must_not_set=False)

    def pass_id(self, field_name='current_pass_id'):
        return self.get_field_with_type(field_name=field_name, tp=int)

    def increase_pass_id(self, field_name='current_pass_id'):
        self.set_field_with_type(
            field_name=field_name,
            value=self.pass_id(field_name=field_name) + 1,
            tp=int,
            must_not_set=False)

    def get_field(self, field_name):
        field = getattr(self.real_context, field_name, None)
        return field

    def get_field_with_type(self, field_name, tp):
        field = self.get_field(field_name)
        assert isinstance(field,
                          tp), "Field %s with type %s, should with type %s" % (
                              field_name, type(field), tp)
        return field

    def in_args(self, field_name='in_args'):
        """

        :param field_name:
        :return:
        :rtype: api.Arguments
        """
        return self.get_field_with_type(field_name=field_name, tp=api.Arguments)

    def set_in_args(self, in_args, field_name='in_args'):
        self.set_field_with_type(
            field_name=field_name,
            value=in_args,
            tp=api.Arguments,
            must_not_set=False)


class BaseRunnerItem(RunnerItem):
    """
    :type context: ContextWrapper
    """

    def __init__(self):
        super(BaseRunnerItem, self).__init__()
        self.context = None

    def store_context(self, context):
        self.context = ContextWrapper(context=context)
