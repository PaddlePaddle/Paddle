import paddle.v2 as paddle
import numpy as np

# init paddle
paddle.init(use_gpu=False)

# network config
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
cost = paddle.layer.mse_cost(input=y_predict, label=y)

# create parameters
parameters = paddle.parameters.create(cost)
# create optimizer
optimizer = paddle.optimizer.Momentum(momentum=0)
# create trainer
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)


# event_handler to print training info
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)


# define training dataset reader
def train_reader():
    train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
    train_y = np.array([-2, -3, -7, -7])

    def reader():
        for i in xrange(train_y.shape[0]):
            yield train_x[i], train_y[i]

    return reader


# define feeding map
feeding = {'x': 0, 'y': 1}

# training
trainer.train(
    reader=paddle.batch(
        train_reader(), batch_size=1),
    feeding=feeding,
    event_handler=event_handler,
    num_passes=100)
