"""
Runner Base Classes.

We can invoke a runner pass by pass. In one pass, Runner will handle everything,
 invoke them batch by batch.


Runner class helps us to extract complex logic in `Trainer.cpp`. Runner contains
several RunnerItem; each RunnerChainItem will do some aspect of Trainer
logic.

For example, the trainer logic could be:

..  code-block: python

    gradient_machine.startPass()
    updater.startPass()
    for each_batch in data:
        gradient_machine.startBatch()
        updater.startBatch()

        gradient_machine.train()

        updater.finishBatch()
        gradient_machine.finishBatch()
    updater.finishPass()
    gradient_machine.finishPass()

We can extract this logic into two RunnerChainItems. GradientMachineOperations
and UpdaterOperations. It just like a middleware framework.
"""

import functools

__all__ = ['RunnerItem', 'Runner', 'RunnerContext']


class RunnerContext(object):
    pass


class RunnerItem(object):
    """
    RunnerItem is an item in Runner. Runner will composite the
    RunnerItems together and invoke the first RunnerChainItem's methods.
    And Runner will pass the next chain item's method as `next_callback`.
    If current chain item is the last item. A default next_callback will be
    passed.

    Context is a global object shared by items.
    """

    def __init__(self):
        pass

    def initialize(self, context, next_callback):
        """
        initialize method. It will be invoked when Runner start to run.

        :param context: a global object shared by items.
        :type context: RunnerContext
        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: None
        :rtype: None
        """
        next_callback(context)

    def finalize(self, next_callback):
        """
        Finalize method. It will be invoked when Runner complete run, and clean
        some state in RunnerItem.

        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: None
        :rtype: None
        """
        next_callback()

    def on_pass_begin(self, next_callback):
        """
        Pass Begin Method. Invoked when a pass begins.

        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: None
        :rtype: None
        """

        next_callback()

    def on_pass_end(self, next_callback):
        """
        Pass End Method. Invoked when a pass ends.

        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: None
        :rtype: None
        """
        next_callback()

    def on_batch_begin(self, next_callback):
        """
        Batch Begin Method. Invoked when a batch begins. Return true if there is
        no more batch could be processed.

        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: True if no more batch could be processed.
        :rtype: bool
        """
        return next_callback()

    def on_batch_end(self, next_callback):
        """
        Batch End Method. Invoked when a batch ends. Return true if there is
        no more batch could be processed.

        :param next_callback: next item's initialize method.
        :type next_callback: callable
        :return: True if no more batch could be processed.
        :rtype: bool
        """
        return next_callback()


def default_next_callback(*args, **kwargs):
    """
    Default next_callback for the last RunnerItem.
    """
    return False


class Runner(object):
    """
    Runner contains many RunnerItem. Each item will do some aspect of
    Trainer/Tester job. Basic usage is shown as below.

    ..  code-block: python

        runner = Runner()

        runner.add_item(ItemA())
        runner.add_item(ItemB())

        with runner:
            runner.run_one_pass()
    """

    # Because Runner is heavily used, so explicit declare the __slots__ for
    # faster attribute access.
    __slots__ = [
        '__init__', 'add_item', '__initialize__', 'run_one_pass', '__enter__',
        '__exit__', '__items__', '__begin_pass__', '__end_pass__',
        '__begin_batch__', '__end_batch__', 'finalize', '__context__'
    ]

    def __init__(self):
        self.__items__ = []
        self.__begin_pass__ = None
        self.__end_pass__ = None
        self.__begin_batch__ = None
        self.__end_batch__ = None
        self.finalize = None

        self.__context__ = RunnerContext()
        self.__context__.runner = self

    def add_item(self, item):
        """
        Add a runner item to runner.
        """
        assert isinstance(item, RunnerItem)
        self.__items__.append(item)

    def __initialize__(self, parent=None):
        if None not in [
                self.__begin_pass__, self.__end_pass__, self.__begin_batch__,
                self.__end_batch__, self.finalize
        ]:
            return False
        else:
            assert len(self.__items__) != 0
            actual_init = default_next_callback
            self.__begin_pass__ = default_next_callback
            self.__end_pass__ = default_next_callback
            self.__begin_batch__ = default_next_callback
            self.__end_batch__ = default_next_callback
            self.finalize = default_next_callback

            for chain in reversed(self.__items__):
                assert isinstance(chain, RunnerItem)
                actual_init = functools.partial(
                    chain.initialize, next_callback=actual_init)
                self.__begin_pass__ = functools.partial(
                    chain.on_pass_begin, next_callback=self.__begin_pass__)
                self.__end_pass__ = functools.partial(
                    chain.on_pass_end, next_callback=self.__end_pass__)
                self.__begin_batch__ = functools.partial(
                    chain.on_batch_begin, next_callback=self.__begin_batch__)
                self.__end_batch__ = functools.partial(
                    chain.on_batch_end, next_callback=self.__end_batch__)
                self.finalize = functools.partial(
                    chain.finalize, next_callback=self.finalize)

            if parent is not None:
                self.__context__.parent = parent

            actual_init(self.__context__)
            return True

    def run_one_pass(self):
        """
        Run one pass for runner. The parent argument will passed to context.
        """

        self.__begin_pass__()
        exit_flag = False
        while not exit_flag:
            exit_flag = self.__begin_batch__()
            if exit_flag:
                break
            exit_flag = self.__end_batch__()
        self.__end_pass__()
        return self.__context__

    def __enter__(self):
        """
        Implementation for with block.
        :return:
        """
        self.__initialize__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Implementation for with block.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.finalize()
