# PaddlePaddle Fluid Source Code Overview

Examples: https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/v2/fluid/tests/book

Core: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/framework

Operator: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/operators

Optimizer: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/optimizer

Memory: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/memory

# Compile Time

The following **defines** the NN. The definition goes into this [protocol buffer](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto).

```python
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

y_predict = fluid.layers.fc(input=x, size=1, act=None)
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(x=cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)
```

- Variables: `x`,  `y`, `y_predict`, `cost` and `avg_cost`. [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/framework.py#L93)
- Layers: `fluid.layers.data`, `fluid.layers.fc` and `fluid.layers.mean` are layers. [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/layers.py)
  - Every Layer has one or more operators and variables/parameters
    - All the operators are defined at [`paddle/operators/`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/operators). Other worth-looking files:
      - Base class: [`paddle/framework/operator.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/operator.h)
      - Operator Registration: [`paddle/framework/op_registry.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/op_registry.h) 
      - Operator Lookup: [`paddle/framework/op_info.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/op_info.h)
- Optimizer: `fluid.optimizer.SGD`. It does the following
  - Add backward operators. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/backward.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/backward.cc)]
  - Add optimizer operators. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/optimizer.py), [C++](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/optimizer)]

# Run Time

The following **evaluates** the NN. Instantiates all the variables, operators.

```python
place = fluid.CPUPlace()
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe = fluid.Executor(place)

# Allocate memory. Initialize Parameter.
exe.run(fluid.default_startup_program())

# Allocate memory. Do computation.
exe.run(fluid.default_main_program(),
        feed=feeder.feed(data),
        fetch_list=[avg_cost])
```

- Place: `place`. one of CPU, GPU or FPGA. [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h)
  - The device handle are at [paddle/platform/device_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/device_context.h)
- Executor: `fluid.Executor(place)`. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/executor.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/executor.cc)]
  - Feeds the data: `feed=feeder.feed(data)`
  - Evaluates all the operators
  - Fetches the result: `fetch_list=[avg_cost]`
- Other worth looking files:
  - Scope: [paddle/framework/scope.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/scope.h). Where all the variables live
    - Variable: [paddle/framework/variable.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/variable.h). Where all the data (most likely tensors) live
      - Tensor: [paddle/framework/tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/tensor.h). Where we allocate memory through [`paddle/memory/`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/memory)
