# Design Doc: The C++ Class `Parameters`

`Parameters` is a concept we designed in PaddlePaddle V2 API. `Parameters` is a container of parameters, which makes PaddlePaddle capable of  sharing parameter between topologies. We described usages of `Parameter` in [api.md](./api.md).

We used Python to implement Parameters when designing V2 API before. There are several defects for the current implementation:
* We just use `memcpy` to share Parameters between topologies, but this is very inefficient. 
* We did not support sharing Parameters while training. We just trigger `memcpy` when start training.

It is necessary that we implement Parameters in CPP side. However, it could result a code refactoring for PaddlePaddle, because PaddlePaddle was designed for training only one topology before, i.e., each GradientMachine contains its Parameter as a data member. In current PaddlePaddle implementation, there are three concepts associated with `Parameters`:

1. `paddle::Parameter`. A `Parameters` is a container for `paddle::Parameter`.
It is evident that we should use `paddle::Parameter` when developing `Parameters`.
However, the `Parameter` class contains many functions and does not have a clear interface.
It contains `create/store Parameter`, `serialize/deserialize`, `optimize(i.e SGD)`, `randomize/zero`.
When we developing `Parameters`, we only use `create/store Parameter` functionality.
We should extract functionalities of Parameter into many classes to clean PaddlePaddle CPP implementation.

2. `paddle::GradientMachine` and its sub-classes, e.g., `paddle::MultiGradientMachine`, `paddle::NeuralNetwork`.
We should pass `Parameters` to `paddle::GradientMachine` when `forward/backward` to avoid `memcpy` between topologies.
Also, we should handle multi-GPU/CPU training, because `forward` and `backward` would perform on multi-GPUs and multi-CPUs.
`Parameters` should dispatch the parameter value to each device, and gather the parameter gradient from each device.

3. `paddle::ParameterUpdater`. The ParameterUpdater is used to update parameters in Paddle. 
So `Parameters` should be used by `paddle::ParameterUpdater`, and `paddle::ParameterUpdater` should optimize `Parameters` (by SGD).


The step by step approach for implementation Parameters in PaddlePaddle C++ core is listed below. Each step should be a PR and could be merged into PaddlePaddle one by one.

1. Clean `paddle::Parameter` interface. Extract the functionalities of `paddle::Parameter` to prepare for the implementation of Parameters.

2. Implementation a `Parameters` class. It just stores the `paddle::Parameter` inside. Make `GradientMachine` uses `Parameters` as a class member.

3. Make `Parameters` support Multi-CPU and Multi-GPU training to prepare for sharing `Parameter` between topologies.
Because we need share `Parameters` between topologies, it is `Parameters`'s response to exchange Parameters between GPUs.
`GradientMachine` should not handle how to exchange Parameters because `GradientMachine` only used to train one topology and we need to support train many topologies in Paddle, i.e., there could be many GradientMachines use one `Parameters`.
   * We should use a global function to exchange Parameters between GPUs, not a member function in `Parameters`. The `MultiGradientMachine` invoke this function, which uses `Parameters` as this function inputs.
   * The MultiGradientMachine contains many functionalities. Extracting the Parameters exchanging logic could make MultiGradientMachine clearer and simpler.

4. Make `Parameters` as an argument for `forward/backward` function, not a data member for `GradientMachine`. For example, `forward` could be `forward(const Parameters& params, ...)` and `backward` could be `backward(Parameters* params, ...)`. After this step, Paddle could share `Parameters` between topologies.

5. `ParameterUpdater` is invoked by `GradientMachine` and `Trainer`, but it updates `Parameters`. In the end of this code refactoring, we could change `ParameterUpdater` directly uses `Parameters` to make `ParameterUpdater`'s implementation clear.
