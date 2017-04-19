"""
Module Trainer
"""
import collections

import py_paddle.swig_paddle as api

from data_feeder import DataFeeder
from topology import Topology
from . import event as v2_event
from . import optimizer as v2_optimizer
from . import parameters as v2_parameters

__all__ = ['SGD']


def default_event_handler(event):
    """
    Default event handler. It will print some log and save mode.

    TODO(yuyang18): Complete it!
    :param event:
    :return:
    """
    pass


class SGD(object):
    """
    Simple SGD Trainer.
    SGD Trainer combines data reader, network topolopy and update_equation together
    to train/test a neural network.

    :param update_equation: The optimizer object.
    :type update_equation: paddle.v2.optimizer.Optimizer
    :param cost: Target cost that neural network should be optimized.
    :type cost: paddle.v2.config_base.Layer
    :param parameters: The parameters dictionary.
    :type parameters: paddle.v2.parameters.Parameters
    :param extra_layers: Some layers in the neural network graph are not
                         in the path of cost layer.
    :type extra_layers: paddle.v2.config_base.Layer
    """

    def __init__(self, cost, parameters, update_equation, extra_layers=None):

        if not isinstance(parameters, v2_parameters.Parameters):
            raise TypeError('parameters should be parameters')

        if not isinstance(update_equation, v2_optimizer.Optimizer):
            raise TypeError("update equation parameter must be "
                            "paddle.v2.optimizer.Optimizer")
        topology = Topology(cost, extra_layers=extra_layers)
        self.__optimizer__ = update_equation
        self.__topology__ = topology
        self.__parameters__ = parameters
        self.__topology_in_proto__ = topology.proto()

        # In local mode, disable sparse_remote_update.
        for param in self.__topology_in_proto__.parameters:
            if param.sparse_remote_update:
                param.sparse_remote_update = False

        self.__data_types__ = topology.data_type()
        gm = api.GradientMachine.createFromConfigProto(
            self.__topology_in_proto__, api.CREATE_MODE_NORMAL,
            self.__optimizer__.enable_types())
        assert isinstance(gm, api.GradientMachine)
        self.__gradient_machine__ = gm
        self.__gradient_machine__.randParameters()
        parameters.append_gradient_machine(gm)

    def train(self, reader, num_passes=1, event_handler=None, feeding=None):
        """
        Training method. Will train num_passes of input data.

        :param reader: A reader that reads and yeilds data items. Usually we use a
                       batched reader to do mini-batch training.
        :type reader: collections.Iterable
        :param num_passes: The total train passes.
        :param event_handler: Event handler. A method will be invoked when event
                              occurred.
        :type event_handler: (BaseEvent) => None
        :param feeding: Feeding is a map of neural network input name and array
                        index that reader returns.
        :type feeding: dict|list
        :return:
        """
        if event_handler is None:
            event_handler = default_event_handler
        __check_train_args__(**locals())

        updater = self.__optimizer__.create_local_updater()
        updater.init(self.__gradient_machine__)

        self.__gradient_machine__.start()
        batch_evaluator = self.__gradient_machine__.makeEvaluator()
        assert isinstance(batch_evaluator, api.Evaluator)
        pass_evaluator = self.__gradient_machine__.makeEvaluator()
        assert isinstance(pass_evaluator, api.Evaluator)
        out_args = api.Arguments.createArguments(0)
        feeder = DataFeeder(self.__data_types__, feeding)
        for pass_id in xrange(num_passes):
            event_handler(v2_event.BeginPass(pass_id))
            pass_evaluator.start()
            updater.startPass()
            for batch_id, data_batch in enumerate(reader()):
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
                cost_sum = out_args.sum()
                cost = cost_sum / len(data_batch)
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

    def test(self, reader, feeding=None):
        """
        Testing method. Will test input data.

        :param reader: A reader that reads and yeilds data items.
        :type reader: collections.Iterable  
        :param feeding: Feeding is a map of neural network input name and array
                        index that reader returns.
        :type feeding: dict
        :return:
        """
        feeder = DataFeeder(self.__data_types__, feeding)
        evaluator = self.__gradient_machine__.makeEvaluator()
        out_args = api.Arguments.createArguments(0)
        evaluator.start()
        total_cost = 0
        num_samples = 0.0
        for data_batch in reader():
            num_samples += len(data_batch)
            self.__gradient_machine__.forward(
                feeder(data_batch), out_args, api.PASS_TEST)
            total_cost += out_args.sum()
            self.__gradient_machine__.eval(evaluator)

        evaluator.finish()
        return v2_event.TestResult(
            evaluator=evaluator, cost=total_cost / num_samples)


def __check_train_args__(reader, event_handler, **kwargs):
    """
    Check train function's argument types
    """
    if not callable(reader) or not isinstance(reader(), collections.Iterator):
        raise TypeError('train_data_reader should be a function, '
                        'which can return a iterator')
    if not callable(event_handler):
        raise TypeError('event handler should be a function')
