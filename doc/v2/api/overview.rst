V2 API Overview
================

The PaddlePaddle V2 API is designed to provide a modern user interface for PaddlePaddle V1(the original layer-based platform of PaddlePaddle),
it proposes some high-level concepts such as `Layers <http://www.paddlepaddle.org/docs/develop/api/en/v2/config/layer.html>`_ , `Optimizer <http://www.paddlepaddle.org/docs/develop/api/en/v2/config/optimizer.html>`_ , `Evaluator <http://www.paddlepaddle.org/docs/develop/api/en/v2/config/evaluators.html>`_  and `Data Reader <http://www.paddlepaddle.org/docs/develop/api/en/v2/data/data_reader.html>`_ to make the model configuration more familiar to users.

A model is composed of the computation described by a group of `Layers`, with `Evaluator` to define the error, `Optimizer` to update the parameters and `Data Reader` to feed in the data.

We also provide the `interface for Training and Inference <http://www.paddlepaddle.org/docs/develop/api/en/v2/run_logic.html>`_ to help control the training and inference phrase,
it has several easy to use methods to better expose the internal running details, different `events <http://www.paddlepaddle.org/docs/develop/api/en/v2/run_logic.html#event>`_ are available to users by writing some callbacks.

All in all, the V2 API gives a higher abstraction and make PaddlePaddle programs require fiew lines of code.
