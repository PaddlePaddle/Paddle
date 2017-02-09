___all__ = ['Optimizer']


class BaseObserveEvent(object):
    def __init__(self, pass_id, batch_id):
        self.__pass_id__ = pass_id
        self.__batch_id__ = batch_id


class TrainOneBatchEvent(BaseObserveEvent):
    def __init__(self, pass_id, batch_id, loss):
        super(TrainOneBatchEvent, self).__init__(pass_id, batch_id)
        self.__loss__ = loss

    def __str__(self):
        return "Pass %d, Batch %d, Loss %f" % (self.__pass_id__,
                                               self.__batch_id__, self.__loss__)


def default_observe_callback(gradient_machine, event):
    if isinstance(event, TrainOneBatchEvent):
        print event
    else:
        raise NotImplementedError()


class Optimizer(object):
    def __init__(self, gradient_machine):
        raise NotImplementedError()

    def train(self, train_reader=None, test_reader=None, observe_callback=None):
        assert hasattr(train_reader, 'next')
        assert test_reader is None or hasattr(test_reader, 'next')
        if observe_callback is None:
            observe_callback = default_observe_callback
        assert callable(observe_callback)
        raise NotImplementedError()
