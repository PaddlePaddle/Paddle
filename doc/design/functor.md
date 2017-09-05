# Design Doc: Operator Functor

PaddlePaddle needs to be fast even we are using `Operator` as our basic building blocks. So we decided to write our kernels into `Functor`s. Each `functor` can be exposed to Python as an `Operator` or can be fused into another huge `functor`. So the `Operator` can be coarse grain without copy-paste code since we can combine many fine grain `functor`s.

So we should take care several problems of `functor` when implementation:

1. Our framework should be able to fuse many `functor`s into one CUDA kernel.
	* For example, `tanh` is a rescaled `sigmoid`. If we have a `add_scalar` functor, a `scale` functor, and a `tanh` functor, we should combine `sigmoid` with `sigmoid = (tanh + 1)/2`.

1. `Functor` should be able to read attributes both from an operator and other `Functor`.
	* For example, `scale` functor should be able to read attribute `scale` from `ScaleOp` and from other `functor`s. e.g., the `sigmoid` functor contains a `scale` functor which `scale=0.5`.

1. A `BinaryFunctor` should take care of `broadcast`. For example, a `minus` functor should not only handle a `tensor` minus a `tensor` one by one, but also a `matrix` minus a `vector` row-wisely. The `functor` developer only need to write `a - b`, the framework should do the rest thing.

1. A Functor should be easily converted to an `Operator`.
	* A `Functor` should be convert to `OpKernel` in one line.
	* Some pre-defined `Operator` and `OpProtoMaker` should be added, such as `UnaryOp`, `BinaryOp`.
