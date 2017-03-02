import collections

import py_paddle.swig_paddle as api

from data_feeder import DataFeeder
from topology import Topology
from . import event as v2_event
from . import optimizer as v2_optimizer
from . import parameters as v2_parameters

__all__ = ['ITrainer', 'SGD']


def default_event_handler(event):
    """
    Default event handler. It will print some log and save mode.

    TODO(yuyang18): Complete it!
    :param event:
    :return:
    """
    pass


class ITrainer(object):
    """
    The interface of Trainer. The only exposed method is `train`.
    """

    def train(self, reader, topology, parameters, event_handler=None):
        """
        train method.

        :param reader:
        :param topology:
        :param parameters:
        :param event_handler:
        :return:
        """

        raise NotImplementedError()


class SGD(ITrainer):
    def __init__(self, cost, parameters, update_equation):
        """
        Simple SGD Trainer.

        :param update_equation: The optimizer object.
        :type update_equation: v2_optimizer.Optimizer
        """

        if not isinstance(parameters, v2_parameters.Parameters):
            raise TypeError('parameters should be parameters')

        if not isinstance(update_equation, v2_optimizer.Optimizer):
            raise TypeError("update equation parameter must be "
                            "paddle.v2.optimizer.Optimizer")
        topology = Topology(cost)
        self.__optimizer__ = update_equation
        self.__topology__ = topology
        self.__parameters__ = parameters
        self.__topology_in_proto__ = topology.proto()
        self.__data_types__ = topology.data_type()
        gm = api.GradientMachine.createFromConfigProto(
            self.__topology_in_proto__, api.CREATE_MODE_NORMAL,
            self.__optimizer__.enable_types())
        assert isinstance(gm, api.GradientMachine)
        parameters.append_gradient_machine(gm)
        self.__gradient_machine__ = gm
        self.__gradient_machine__.randParameters()

    def train(self, reader, num_passes=1, event_handler=None, reader_dict=None):
        """
        Training method. Will train num_passes of input data.

        :param reader:
        :param topology: Network Topology, use one or more Layers to represent it.
        :param parameters: The parameter pools.
        :param num_passes: The total train passes.
        :param event_handler: Event handler. A method will be invoked when event
                              occurred.
        :type event_handler: (BaseEvent) => None
        :return:
        """
        if event_handler is None:
            event_handler = default_event_handler

        if reader_dict is None:
            reader_dict = self.default_reader_dict()

        __check_train_args__(**locals())

        updater = self.__optimizer__.create_local_updater()
        updater.init(self.__gradient_machine__)

        self.__gradient_machine__.start()
        batch_evaluator = self.__gradient_machine__.makeEvaluator()
        assert isinstance(batch_evaluator, api.Evaluator)
        pass_evaluator = self.__gradient_machine__.makeEvaluator()
        assert isinstance(pass_evaluator, api.Evaluator)
        out_args = api.Arguments.createArguments(0)

        feeder = DataFeeder(self.__data_types__, reader_dict)

        for pass_id in xrange(num_passes):
            event_handler(v2_event.BeginPass(pass_id))
            pass_evaluator.start()
            updater.startPass()
            for batch_id, data_batch in enumerate(reader()):
                pass_type = updater.startBatch(len(data_batch))
                self.__gradient_machine__.forwardBackward(
                    feeder(data_batch), out_args, pass_type)
                batch_evaluator.start()
                event_handler(
                    v2_event.BeginIteration(
                        pass_id=pass_id, batch_id=batch_id))
                pass_type = updater.startBatch(len(data_batch))
                self.__gradient_machine__.forwardBackward(
                    feeder(data_batch), out_args, pass_type)
                self.__gradient_machine__.eval(pass_evaluator)
                self.__gradient_machine__.eval(batch_evaluator)
                for each_param in self.__gradient_machine__.getNonStaticParameters(
                ):
                    updater.update(each_param)
                # Get cost. We use numpy to calculate total cost for this batch.
                cost_vec = out_args.getSlotValue(0)
                cost_vec = cost_vec.copyToNumpyMat()
                cost = cost_vec.sum() / len(data_batch)
                updater.finishBatch(cost)
                batch_evaluator.finish()
                event_handler(
                    v2_event.EndIteration(
                        pass_id=pass_id,
                        batch_id=batch_id,
                        cost=cost,
                        evaluator=batch_evaluator))

            updater.finishPass()
            pass_evaluator.finish()
            event_handler(v2_event.EndPass(pass_id, evaluator=pass_evaluator))
        self.__gradient_machine__.finish()

    def default_reader_dict(self):
        reader_dict = dict()
        for i, tp in enumerate(self.__data_types__):
            reader_dict[tp[0]] = i
        return reader_dict

    def test(self, reader, reader_dict=None):
        if reader_dict is None:
            reader_dict = self.default_reader_dict()

        feeder = DataFeeder(self.__data_types__, reader_dict)
        evaluator = self.__gradient_machine__.makeEvaluator()
        out_args = api.Arguments.createArguments(0)
        evaluator.start()
        for data_batch in reader():
            self.__gradient_machine__.forward(
                feeder(data_batch), out_args, api.PASS_TEST)
            self.__gradient_machine__.eval(evaluator)

        evaluator.finish()
        return v2_event.TestResult(evaluator=evaluator)


def __check_train_args__(reader, event_handler, **kwargs):
    """
    Check train function's argument types
    """
    if not callable(reader) or not isinstance(reader(), collections.Iterator):
        raise TypeError('train_data_reader should be a function, '
                        'which can return a iterator')
    if not callable(event_handler):
        raise TypeError('event handler should be a function')
