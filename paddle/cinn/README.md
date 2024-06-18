```
                        ___                    ___          ___
                       /\__\                  /\  \        /\  \
                      /:/  /       ___        \:\  \       \:\  \
                     /:/  /       /\__\        \:\  \       \:\  \
                    /:/  /  ___  /:/__/    _____\:\  \  _____\:\  \
                   /:/__/  /\__\/::\  \   /::::::::\__\/::::::::\__\
                   \:\  \ /:/  /\/\:\  \__\:\~~\~~\/__/\:\~~\~~\/__/
                    \:\  /:/  /    \:\/\__\\:\  \       \:\  \
                     \:\/:/  /      \::/  / \:\  \       \:\  \
                      \::/  /       /:/  /   \:\__\       \:\__\
                       \/__/        \/__/     \/__/        \/__/

```


# CINN : Compiler Infrastructure for Neural Networks

The project CINN is a machine learning compiler and executor for multiple hardware backends.
It is designed to provide multiple layers of APIs to make tensor computation easier to define,  faster to execute, and more convenient to extend with hardware backends.
Currently, it targets x86 CPUs and Nvidia GPUs.

This project is under active development.

## How it works

The CINN lowers a traditional DNN model into a two-level intermediate representation(IR), the high-level IR(HLIR) and CINN IR.

The HLIR helps to define some domain-specific computation and perform some overall optimization on the IR-graph;
the CINN IR helps to represent some computation semantic and finally lower to a hardware backend.

Both levels of IR have the similar SSA graph, analysis and optimization facilities.
The schedule transform is applied on the CINN IR to do optimizations.

For more details, you can refer to:
https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/cinn

##  Getting Started

### Compile

Clone PaddlePaddle first.

```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
```

Build paddle with cinn:

```
cmake .. -DWITH_CINN=ON -DWITH_GPU=ON
```

And then

```
make -j
```

### Install

Install paddle with cinn:

```
pip install python/dist/paddlepaddle_gpu-xxx.whl
```

Install cinn only:

```
pip install python/dist/cinn_gpu-xxx.whl
```

Then you can import paddle in the python environment and check if a paddle version with CINN is installed.

```
import paddle
paddle.is_compiled_with_cinn()
```

### Concepts

There are two levels of APIs in CINN, the higher level is HLIR and the lower level is CINN IR, both contain some concepts.

In HLIR

- `frontend::Program`, the program helps to define a machine learning computation,
- `hlir::framework::Tensor`, multi-dimensional arrays helps to manage a memory buffer.
- `hlir::framework::Program`, the final executable program in runtime. It holds many basic executable elements.
- `hlir::framework::Graph`, the graph that represents the structure of a model. Each node in the graph represents an operator (conv2d, relu, mul, etc.).
- `hlir::framework::GraphCompiler`, the compiler that transforms the graph representation(hlir::framework::Graph) of a model into an executable program(hlir::framework::Program).

In CINN IR

- `Compute`, the method to define a computation,
- `Lower`, the method to lower a computation to the corresponding IR,
- `LoweredFunc`, the function defined in CINN IR,
- `Var`, a scalar variable,
- `Expr`, an expression represents any CINN IR node(no specified Statement node),

## License

CINN is licensed under the [Apache 2.0 license](LICENSE).

## Acknowledgement

CINN learned a lot from the following projects:

- [Halide](https://github.com/halide/Halide): Referenced the design of most IR nodes,
- [TVM](https://github.com/apache/tvm): We learned many ideas including the semantics of some schedule primitives, TOPI, NNVM, and so on,
- [tiramisu](https://github.com/Tiramisu-Compiler): The isl usage, polyhedral compilation, schedule primitive implementation, and so on,
- [tensorflow/xla](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla): Referenced the semantics of the primitive operations.
