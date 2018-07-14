import paddle
import paddle.fluid as fluid
import numpy

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE)


def train_program():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.001)


trainer = fluid.Trainer(
    train_func=train_program,
    place=fluid.CPUPlace(),
    optimizer_func=optimizer_program)

feed_order = ['x', 'y']

# Specify the directory path to save the parameters
params_dirname = "fit_a_line.inference.model"

step = 0

# event_handler to print training and testing info
def event_handler(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 10 == 0:  # every 10 batches, record a test cost
            test_metrics = trainer.test(
                reader=test_reader, feed_order=feed_order)

            if test_metrics[0] < 10.0:
                # If the accuracy is good enough, we can stop the training.
                print('loss is less than 10.0, stop')
                trainer.stop()

        # We can save the trained parameters for the inferences later
        if params_dirname is not None:
            trainer.save_params(params_dirname)

        step += 1


# The training could take up to a few minutes.
import pdb;pdb.set_trace()
trainer.train(
    reader=train_reader,
    num_epochs=100,
    event_handler=event_handler,
    feed_order=feed_order)


def inference_program():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    return y_predict


inferencer = fluid.Inferencer(
    infer_func=inference_program, param_path=params_dirname, place=fluid.CPUPlace())

batch_size = 10
# tensor_x = numpy.random.uniform(0, 10, [batch_size, 13]).astype("float32")
tensor_x = numpy.array([[0.00632,  18.00,   2.310,  0,  0.5380,  6.5750,  65.20,  4.0900,   1,  296.0,  15.30, 396.90,   4.98],
                        [0.02731,   0.00,   7.070,  0,  0.4690,  6.4210,  78.90,  4.9671,   2,  242.0,  17.80, 396.90,   9.14]]).astype("float32")

'''
root@9ebe6d3b3b6b:/work/Paddle/python/paddle/fluid/tests/book/high-level-api/fit_a_line# python book_fit_a_line.py 
> /work/Paddle/python/paddle/fluid/tests/book/high-level-api/fit_a_line/book_fit_a_line.py(67)<module>()
-> trainer.train(
(Pdb) c
('infer results: ', array([[-323.3439 ],
       [-264.91995]], dtype=float32))
root@9ebe6d3b3b6b:/work/Paddle/python/paddle/fluid/tests/book/high-level-api/fit_a_line# python book_fit_a_line.py 
> /work/Paddle/python/paddle/fluid/tests/book/high-level-api/fit_a_line/book_fit_a_line.py(67)<module>()
-> trainer.train(
(Pdb) c
('infer results: ', array([[-605.6774],
       [-529.0045]], dtype=float32))
'''

results = inferencer.infer({'x': tensor_x})
print("infer results: ", results[0])
