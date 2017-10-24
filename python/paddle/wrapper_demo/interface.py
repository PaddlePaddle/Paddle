import paddle.v2 as pd
import paddle.v2.framework.core as core
from util import Graph

__all__ = [
    "Variable",
]



class Layer(object):
    pass


def eval(fetches=[], block=global_block):
    '''
    fetches: list
        variables that user want to fetches the latest value.

    It will help to run operators help to eval specific target variables. In details,
    it will build a `Block` and put related operators in it, execute and return
    all the python value of variables in the fetches list.

    For example:
        def data_provider(path):
            # ...
            yield batch

        # data slots
        image = pd.data('image')
        label = pd.data('label')

        # model config
        fc_out = pd.fc(image, size=128)
        prediction = pd.softmax(fc_out, size=10)
        cost = pd.cross_entropy(label, prediction)
        loss = pd.SGDOptimizer([cost])

        # train
        for batch_id, batch in enumerate(data_provider("./data.txt")):
            _loss, _cost = pd.eval([loss, eval],
                                   feed_data={'image': xxx, 'label':xxx})
            print 'batch %d cost %f" % (batch_id, _cost)

        # infer
        _prediction = pd.eval([prediction], feed_data={'image': xxx})
    '''
    graph = Graph(block.ops)
    local_block = Block()
    for op in graph.reverse_dfs(endpoints=fetches):
        local_block.append(op)
    local_block.execute()
    return [var.py_value() for var in fetches]


def variable_initializer(vars=[]):
    '''
    run variable's initializer ops and return
    '''
    pass
