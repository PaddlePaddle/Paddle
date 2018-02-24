# V2 API Overview

The PaddlePaddle V2 API is designed to provide a modern user interface for PaddlePaddle V1(the original layer-based platform of PaddlePaddle), it proposes some high-level concepts such as [Layers](http://www.paddlepaddle.org/docs/develop/api/en/v2/config/layer.html),[Optimizer](http://www.paddlepaddle.org/docs/develop/api/en/v2/config/optimizer.html),[Evaluator](http://www.paddlepaddle.org/docs/develop/api/en/v2/config/evaluators.html) and [Data Reader](http://www.paddlepaddle.org/docs/develop/api/en/v2/data/data_reader.html) to make the model configuration more familiar to users.

A model is composed of the computation described by a group of `Layers`, with `Evaluator` to define the error, `Optimizer` to update the parameters and `Data Reader` to feed in the data.

We also provide the [interface for Training and Inference](http://www.paddlepaddle.org/docs/develop/api/en/v2/run_logic.html) to help control the training and inference phrase, it has several easy to use methods

- `paddle.train` 
- `paddle.test`
- `paddle.infer`

to better expose the internal running details, different [Events](http://www.paddlepaddle.org/docs/develop/api/en/v2/run_logic.html#event) are available to users by writing some callbacks.
