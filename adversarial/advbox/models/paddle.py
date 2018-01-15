from __future__ import absolute_import

import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from paddle.v2.fluid.framework import program_guard

from .base import Model


class PaddleModel(Model):
    """
    Create a PaddleModel instance.
    When you need to generate a adversarial sample, you should construct an instance of PaddleModel.

    Args:
        program(paddle.v2.fluid.framework.Program): The program of the model which generate the adversarial sample.
        input_name(string): The name of the input.
        logits_name(string): The name of the logits.
        predict_name(string): The name of the predict.
        cost_name(string): The name of the loss in the program.
    """

    def __init__(self,
                 program,
                 input_name,
                 logits_name,
                 predict_name,
                 cost_name,
                 bounds,
                 channel_axis=3,
                 preprocess=None):
        super(PaddleModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)

        if preprocess is None:
            preprocess = (0, 1)

        self._program = program
        self._place = fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)

        self._input_name = input_name
        self._logits_name = logits_name
        self._predict_name = predict_name
        self._cost_name = cost_name

        # gradient
        loss = self._program.block(0).var(self._cost_name)
        param_grads = fluid.backward.append_backward(
            loss, parameter_list=[self._input_name])
        self._gradient = dict(param_grads)[self._input_name]

    def predict(self, image_batch):
        """
            Predict the label of the image_batch.

            Args:
                image_batch(list): The image and label tuple list.
            Return:
                numpy.ndarray: predictions of the images with shape (batch_size, num_of_classes).
        """
        feeder = fluid.DataFeeder(
            feed_list=[self._input_name, self._logits_name],
            place=self._place,
            program=self._program)
        predict_var = self._program.block(0).var(self._predict_name)
        predict = self._exe.run(self._program,
                                feed=feeder.feed(image_batch),
                                fetch_list=[predict_var])
        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label. 

        Return:
            int: the number of classes
        """
        predict_var = self._program.block(0).var(self._predict_name)
        assert len(predict_var.shape) == 2
        return predict_var.shape[1]

    def gradient(self, image_batch):
        """
        Calculate the gradient of the loss w.r.t the input.

        Args:
            image_batch(list): The image and label tuple list.
        Return:
            list: The list of the gradient of the image.
        """
        feeder = fluid.DataFeeder(
            feed_list=[self._input_name, self._logits_name],
            place=self._place,
            program=self._program)

        grad, = self._exe.run(self._program,
                              feed=feeder.feed(image_batch),
                              fetch_list=[self._gradient])
        return grad
