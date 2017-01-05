from .base import *
from .items import *


class RunnerBuilder(object):
    def __init__(self, network, use_gpu=False, device_count=1):
        self.__runner__ = Runner()
        self.__runner__.add_item(Counter())
        self.__network__ = network
        self.__runner__.add_item(
            set_device(
                use_gpu=use_gpu, device_count=device_count))
        self.__runner__.add_item(
            CreateGradientMachine(network=self.__network__))

        self.__train_data__ = None
        self.__updater__ = None
        self.__gradient_machine__ = None
        self.__init_param__ = None
        self.__evaluate__ = []

    def with_std_random_init_params(self):
        self.__init_param__ = std_random_init_params()
        return self

    def with_train_data(self, method, file_list=None, batch_size=None,
                        **kwargs):
        if batch_size is None:
            batch_size = self.__network__.optimize_graph().batch_size

        if file_list is None:
            file_list = [None]

        self.__train_data__ = BasicPaddleTrainerDataProvider(
            network=self.__network__,
            method=method,
            file_list=file_list,
            batch_size=batch_size,
            **kwargs)
        return self

    def with_std_local_updater(self):
        self.__updater__ = BasicLocalParameterUpdater(network=self.__network__)
        return self

    def with_std_gradient_machine_ops(self):
        self.__gradient_machine__ = BasicGradientMachineTrainOps()
        return self

    def with_batch_evaluator(self, prefix=None):
        self.__evaluate__ = [BatchEvaluate(prefix=prefix)]
        return self

    def with_pass_evaluator(self, prefix=None):
        self.__evaluate__ = [PassEvaluate(prefix=prefix)]
        return self

    def with_std_tester(self, method, file_list, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.__network__.optimize_graph().batch_size

        # tester should be a evaluator, too
        self.__evaluate__.append(
            TestOnPassEnd(
                network=self.__network__,
                method=method,
                file_list=file_list,
                batch_size=batch_size))
        return self

    def with_std_param_saver(self):
        self.__evaluate__.append(SaveParamsOnPassEnd())
        return self

    def with_std_local_trainer(self, **kwargs):
        return self.with_std_random_init_params().with_train_data(
            **kwargs).with_std_local_updater().with_std_gradient_machine_ops(
            ).with_batch_evaluator().with_std_param_saver()

    def with_observer(self, on_batch_end=None, on_pass_end=None):
        self.__evaluate__.append(BaseObserveItem(on_batch_end, on_pass_end))
        return self

    def build(self):
        if self.__init_param__ is None:
            self.with_std_random_init_params()
        self.__runner__.add_item(self.__init_param__)
        self.__runner__.add_item(self.__train_data__)
        if self.__updater__ is None:
            self.with_std_local_updater()
        self.__runner__.add_item(self.__updater__)
        if self.__gradient_machine__ is None:
            self.with_std_gradient_machine_ops()
        self.__runner__.add_item(self.__gradient_machine__)
        for each in self.__evaluate__:
            self.__runner__.add_item(each)
        return self.__runner__
