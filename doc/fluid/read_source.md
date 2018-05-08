# PaddlePaddle Fluid Source Code Overview

Examples: https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/book

Core: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/framework

Operator: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators

Memory: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/memory

Platform: https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/platform

# Compile Time

The following **defines** the NN. The definition goes into this [protocol buffer](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto).

```python
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

y_predict = fluid.layers.fc(input=x, size=1, act=None)
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(x=cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)
```

- Variables: `x`,  `y`, `y_predict`, `cost` and `avg_cost`. [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/framework.py#)
- Layers: `fluid.layers.data`, `fluid.layers.fc` and `fluid.layers.mean` are layers. [Python](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/layers)
  - Every Layer has one or more operators and variables/parameters
    - All the operators are defined at [`paddle/fluid/operators/`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators). Other worth-looking files:
      - Base class: [`paddle/fluid/framework/operator.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h)
      - Operator Registration: [`paddle/fluid/framework/op_registry.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/op_registry.h)
      - Operator Lookup: [`paddle/fluid/framework/op_info.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/op_info.h)
- Optimizer: `fluid.optimizer.SGD`. It does the following
  - Add backward operators. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/backward.py)]
  - Add optimizer operators. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/optimizer.py)]

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

- Place: `place`. one of CPU, GPU or FPGA. [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/place.h)
  - The device handle are at [paddle/fluid/platform/device_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/device_context.h)
- Executor: `fluid.Executor(place)`. [[Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/executor.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/executor.cc)]
  - Feeds the data: `feed=feeder.feed(data)`
  - Evaluates all the operators
  - Fetches the result: `fetch_list=[avg_cost]`
- Other worth looking files:
  - Scope: [paddle/fluid/framework/scope.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/scope.h). Where all the variables live
    - Variable: [paddle/fluid/framework/variable.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/variable.h). Where all the data (most likely tensors) live
      - Tensor: [paddle/fluid/framework/tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/tensor.h). Where we allocate memory through [`paddle/fluid/memory/`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/memory)
