# Parameters in CPP

`Parameters` is a concept we designed in Paddle V2 API. `Parameters` is a container of parameters, and make Paddle can shared parameter between topologies. We described usages of `Parameter` in [api.md](./api.md).

We used Python to implementation Parameters before during API design phase. There are several defects for current implementation:
* We just use `memcpy` to share Parameters between topologies, but this is very inefficient. 
* We did not implement share Parameters while training is not complete. We just trigger `memcpy` when start training.

It is necessary we implement Parameters in CPP side. However, it could be a refactorization for Paddle, because Paddle was designed for training only one topology before. In current Paddle implementation, there are three concepts associated with `Parameters`:

1. `paddle::Parameter`. A `Parameters` is a container for `paddle::Parameter`. It is evident that we should use `paddle::Parameter` when developing `Parameters`. However, the `Parameter` class contains many functions and does not have a clear interface. It contains `create/store Parameter`, `serialize/deserialize`, `optimize(i.e SGD)`, `randomize/zero`. We just need `paddle::Parameter` just create and store `Tensors (or Matrix currently)`.  We should extract functionalities of Parameter into many classes to clean Paddle CPP implementation.
2. `paddle::GradientMachine` and its sub-classes, i.e., `paddle::MultiGradientMachine`, `paddle::NeuralNetwork`. We should pass `Parameters` to `paddle::GradientMachine` when `forward/backward` to avoid `memcpy` between topologies. Also, we should handle multi-GPU/CPU training, because `forward` and `backward` would perform on multi-GPUs and multi-CPUs. `Parameters` should dispatch the parameter value to each device, and gather the parameter gradient from each device.

3. `paddle::ParameterUpdater`. The ParameterUpdater is used to update parameters in Paddle. So `Parameters` should be used by `paddle::ParameterUpdater`, and `paddle::ParameterUpdater` should optimize `Parameters` (by SGD).


The step by step approach for implementation Parameters in Paddle C++ core is listed below. Each step should be a PR merged into Paddle.

1. Clean `paddle::Parameter` interface. Extract the functionalities of `paddle::Parameter` to prepare for the implementation of Parameters.

2. Implementation a `Parameters` class. It just stores the `paddle::Parameter` inside. Make `GradientMachine` uses `Parameters` as a class member.

3. Make `Parameters` support Multi-CPU and Multi-GPU training to prepare for sharing `Parameter` between topologies.  Because we need sharing `Parameter` between topologies, it is `Parameters`'s response to exchange Parameter between GPUs not `GradientMachine`, because `GradientMachine` only used for one topology.

4. Make `Parameters` as an argument for `forward/backward` function, not a data member for `GradientMachine`. For example, `forward` could be `forward(const Parameters& params, ...)` and `backward` could be `backward(Parameters* params, ...)`. After this step, Paddle could share `Parameters` between topologies.

5. `ParameterUpdater` is invoked by `GradientMachine` and `Trainer`, but it updates `Parameters`. In the end, we could change `ParameterUpdater` directly uses `Parameters` to make Paddle implementation clear.
