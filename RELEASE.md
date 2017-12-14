# Release v0.11.0

## PaddlePaddle Fluid

- Release 0.11.0 includes a new feature *PaddlePaddle Fluid*.  Fluid is
  designed to allow users to program like PyTorch and TensorFlow Eager Execution.
  In these systems, there is no longer the concept *model* and applications
  do not include a symbolic description of a graph of operators nor a sequence
  of layers. Instead, applications look exactly like a usual program that
  describes a process of training or inference.  The difference between
  Fluid and PyTorch or Eager Execution is that Fluid doesn't rely on Python's
  control-flow, `if-then-else` nor `for`.  Instead, Fluid provides its
  C++ implementations and their Python binding using the `with` statement.  For an example

  https://github.com/PaddlePaddle/Paddle/blob/3df78ed2a98d37f7ae6725894cc7514effd5664b/python/paddle/v2/fluid/tests/test_while_op.py#L36-L44

- In 0.11.0, we provides a C++ class `Executor` to run a Fluid program.
Executor works like an interpreter. In future version, we will improve
`Executor` into a debugger like GDB, and we might provide some compilers,
which, for example, takes an application like the above one, and outputs
an equivalent C++ source program, which can be compiled using
[`nvcc`](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
to generate binaries that use CUDA, or using
[`icc`](https://software.intel.com/en-us/c-compilers) to generate binaries
that make full use of Intel CPUs.

## New Features

* Release `PaddlePaddle Fluid`.
* Add C-API for model inference
* Use fluid API to create a simple GAN demo.
* Add develop guide about performance tunning.
* Add retry when download `paddle.v2.dataset`.
* Linking protobuf-lite not protobuf in C++. Reduce the binary size.
* Feature [Elastic Deep Learning (EDL)](https://github.com/PaddlePaddle/cloud/tree/develop/doc/autoscale/experiment) released.
* A new style cmake functions for Paddle. It is based on Bazel API.
* Automatically download and compile with Intel® [MKLML](https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz) library as CBLAS when build `WITH_MKL=ON`.
* [Intel® MKL-DNN on PaddlePaddle](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn):
  - Complete 11 MKL-DNN layers: Convolution, Fully connectivity, Pooling, ReLU, Tanh, ELU, Softmax, BatchNorm, AddTo, Concat, LRN.
  - Complete 3 MKL-DNN networks: VGG-19, ResNet-50, GoogleNet
  - [Benchmark](https://github.com/PaddlePaddle/Paddle/blob/develop/benchmark/IntelOptimizedPaddle.md) on Intel Skylake 6148 CPU: 2~3x training speedup compared with MKLML.
* Add the [`softsign` activation](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/activation.html#softsign).
* Add the [dot product layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#dot-prod).
* Add the [L2 distance layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#l2-distance).
* Add the [sub-nested sequence layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#sub-nested-seq).
* Add the [kmax sequence score layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#kmax-sequence-score).
* Add the [sequence slice layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#seq-slice).
* Add the [row convolution layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#row-conv)
* Add mobile friendly webpages.

## Improvements

* Build and install using a single `whl` package.
* [Custom evaluating in V2 API](https://github.com/PaddlePaddle/models/tree/develop/ltr#训练过程中输出自定义评估指标).
* Change `PADDLE_ONLY_CPU` to `PADDLE_WITH_GPU`, since we will support many kinds of devices.
* Remove buggy BarrierStat.
* Clean and remove unused functions in paddle::Parameter.
* Remove ProtoDataProvider.
* Huber loss supports both regression and classification.
* Add the `stride` parameter  for sequence pooling layers.
* Enable v2 API use cudnn batch normalization automatically.
* The BN layer's parameter can be shared by a fixed the parameter name.
* Support variable-dimension input feature for 2D convolution operation.
* Refine cmake about CUDA to automatically detect GPU architecture.
* Improved website navigation.

## Bug Fixes

* Fix bug in ROI pooling. cc9a761
* Fix AUC is zero when label is dense vector. #5274
* Fix bug in WarpCTC layer.

# Release v0.10.0

We are glad to release version 0.10.0.  In this version, we are happy to release the new 
[Python API](http://research.baidu.com/paddlepaddles-new-api-simplifies-deep-learning-programs/).

- Our old Python API is kind of out of date.  It's hard to learn and hard to
  use.  To write a PaddlePaddle program using the old API, we'd have to write
  at least two Python files: one `data provider` and another one that defines
  the network topology.  Users start a PaddlePaddle job by running the
  `paddle_trainer` C++ program, which calls Python interpreter to run the
  network topology configuration script and then start the training loop,
  which iteratively calls the data provider function to load minibatches.
  This prevents us from writing a Python program in a modern way, e.g., in the
  Jupyter Notebook.
  
- The new API, which we often refer to as the *v2 API*, allows us to write
  much shorter Python programs to define the network and the data in a single
  .py file.  Also, this program can run in Jupyter Notebook, since the entry
  point is in Python program and PaddlePaddle runs as a shared library loaded
  and invoked by this Python program.
  
Basing on the new API, we delivered an online interative
book, [Deep Learning 101](http://book.paddlepaddle.org/index.en.html)
and [its Chinese version](http://book.paddlepaddle.org/).

We also worked on updating our online documentation to describe the new API.
But this is an ongoing work.  We will release more documentation improvements
in the next version.

We also worked on bring the new API to distributed model training (via MPI and
Kubernetes).  This work is ongoing. We will release more about it in the next
version.

## New Features

* We release [new Python API](http://research.baidu.com/paddlepaddles-new-api-simplifies-deep-learning-programs/).
* Deep Learning 101 book in [English](http://book.paddlepaddle.org/index.en.html) and [Chinese](http://book.paddlepaddle.org/).
* Support rectangle input for CNN.
* Support stride pooling for seqlastin and seqfirstin.
* Expose `seq_concat_layer/seq_reshape_layer` in `trainer_config_helpers`.
* Add dataset package: CIFAR, MNIST, IMDB, WMT14, CONLL05, movielens, imikolov.
* Add Priorbox layer for Single Shot Multibox Detection. 
* Add smooth L1 cost.
* Add data reader creator and data reader decorator for v2 API.
* Add the CPU implementation of cmrnorm projection.

## Improvements

* Support Python virtualenv for `paddle_trainer`.
* Add pre-commit hooks, used for automatically format our code.
* Upgrade protobuf to version 3.x.
* Add an option to check data type in Python data provider.
* Speedup the backward of average layer on GPU.
* Documentation refinement.
* Check dead links in documents using Travis-CI.
* Add a example for explaining `sparse_vector`.
* Add ReLU in layer_math.py
* Simplify data processing flow for Quick Start.
* Support CUDNN Deconv.
* Add data feeder in v2 API.
* Support predicting the samples from sys.stdin for sentiment demo.
* Provide multi-proccess interface for image preprocessing. 
* Add benchmark document for v1 API.
* Add ReLU in `layer_math.py`.
* Add packages for automatically downloading public datasets.
* Rename `Argument::sumCost` to `Argument::sum` since class `Argument` is nothing with cost.
* Expose Argument::sum to Python
* Add a new `TensorExpression` implementation for matrix-related expression evaluations.
* Add lazy assignment for optimizing the calculation of a batch of multiple expressions.
* Add abstract calss `Function` and its implementation:
  * `PadFunc` and `PadGradFunc`.
  * `ContextProjectionForwardFunc` and `ContextProjectionBackwardFunc`.
  * `CosSimBackward` and `CosSimBackwardFunc`.
  * `CrossMapNormalFunc` and `CrossMapNormalGradFunc`.
  * `MulFunc`.
* Add class `AutoCompare` and `FunctionCompare`, which make it easier to write unit tests for comparing gpu and cpu version of a function.
* Generate `libpaddle_test_main.a` and remove the main function inside the test file.
* Support dense numpy vector in PyDataProvider2.
* Clean code base, remove some copy-n-pasted code snippets:
  * Extract `RowBuffer` class for `SparseRowMatrix`.
  * Clean the  interface of `GradientMachine`.
  * Use `override` keyword in layer.
  * Simplify `Evaluator::create`, use `ClassRegister` to create `Evaluator`s.
* Check MD5 checksum when downloading demo's dataset.
* Add `paddle::Error` which intentially replace `LOG(FATAL)` in Paddle.

## Bug Fixes

* Check layer input types for `recurrent_group`.
* Don't run `clang-format` with .cu source files.
* Fix bugs with `LogActivation`.
* Fix the bug that runs `test_layerHelpers` multiple times.
* Fix the bug that the seq2seq demo exceeds protobuf message size limit.
* Fix the bug in dataprovider converter in GPU mode.
* Fix a bug in `GatedRecurrentLayer`.
* Fix bug for `BatchNorm` when testing more than one models.
* Fix broken unit test of paramRelu.
* Fix some compile-time warnings about `CpuSparseMatrix`.
* Fix `MultiGradientMachine` error when `trainer_count > batch_size`.
* Fix bugs that prevents from asynchronous data loading in `PyDataProvider2`.

# Release v0.9.0

## New Features:

* New Layers
  * bilinear interpolation layer.
  * spatial pyramid-pool layer.
  * de-convolution layer.
  * maxout layer.
* Support rectangle padding, stride, window and input for Pooling Operation.
* Add —job=time in trainer, which can be used to print time info without compiler option -WITH_TIMER=ON.
* Expose cost_weight/nce_layer in `trainer_config_helpers`
* Add FAQ, concepts, h-rnn docs.
* Add Bidi-LSTM and DB-LSTM to quick start demo @alvations
* Add usage track scripts.

## Improvements

* Add Travis-CI for Mac OS X. Enable swig unittest in Travis-CI. Skip Travis-CI when only docs are changed.
* Add code coverage tools.
* Refine convolution layer to speedup and reduce GPU memory.
* Speed up PyDataProvider2
* Add ubuntu deb package build scripts.
* Make Paddle use git-flow branching model.
* PServer support no parameter blocks.

## Bug Fixes

* add zlib link to py_paddle
* add input sparse data check for sparse layer at runtime
* Bug fix for sparse matrix multiplication
* Fix floating-point overflow problem of tanh
* Fix some nvcc compile options
* Fix a bug in yield dictionary in DataProvider
* Fix SRL hang when exit.

# Release v0.8.0beta.1
New features:

* Mac OSX is supported by source code. #138
   * Both GPU and CPU versions of PaddlePaddle are supported.

* Support CUDA 8.0

* Enhance `PyDataProvider2`
   * Add dictionary yield format. `PyDataProvider2` can yield a dictionary with key is data_layer's name, value is features.
   * Add `min_pool_size` to control memory pool in provider.

* Add `deb` install package & docker image for no_avx machines.
   * Especially for cloud computing and virtual machines

* Automatically disable `avx` instructions in cmake when machine's CPU don't support `avx` instructions.

* Add Parallel NN api in trainer_config_helpers.

* Add `travis ci` for Github

Bug fixes:

* Several bugs in trainer_config_helpers. Also complete the unittest for trainer_config_helpers
* Check if PaddlePaddle is installed when unittest.
* Fix bugs in GTX series GPU
* Fix bug in MultinomialSampler

Also more documentation was written since last release.

# Release v0.8.0beta.0

PaddlePaddle v0.8.0beta.0 release. The install package is not stable yet and it's a pre-release version.
