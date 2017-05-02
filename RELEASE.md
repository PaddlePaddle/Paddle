# Release v0.10.0

## New Features

* We release [new python API](http://research.baidu.com/paddlepaddles-new-api-simplifies-deep-learning-programs/).
* Deep Learning 101 book in [English](http://book.paddlepaddle.org/index.en.html) and [Chinese](http://book.paddlepaddle.org/).
* Support rectangle input for CNN.
* Support stride pooling for seqlastin and seqfirstin.
* Expose seq_concat_layer/seq_reshape_layer in `trainer_config_helpers`.
* Add dataset package
  - CIFAR, MNIST, IMDB, WMT14, CONLL05, movielens, imikolov.
* Add Priorbox layer for Single Shot Multibox Detection. 
* Add smooth L1 cost.
* Add data reader creator and data reader decorator for v2 API.
* Add the cpu implementation of cmrnorm-projection.

## Improvements

* Support python virtualenv for `paddle_trainer` process.
* Add pre-commit hooks, used for automatically format our code.
* Use Protobuf 3.X as the default Paddle Protobuf version.
* Add an option to check data type in python data provider.
* Speedup the backward of average layer on GPU.
* Reorganize the catalog of doc/ and refine several docs.
* Add Travis-CI for checking dead links.
* Add a example for explaining sparse_vector.
* Add Relu in layer_math.py
* Simplify data processing flow for quick start.
* Support CUDNN Deconv.
* Add data feeder for v2 API.
* Support predicting the samples from sys.stdin for sentiment demo.
* Provide multi-proccess interface for image preprocessing. 
* Add benchmark document for v1 API.
* Add Relu in layer_math.py.
* Add packages for automatically downloading public datasets.
* Rename Argument::sumCost to Argument::sum since Argument does not have to have any relationship with cost.
  * Expose Argument::sum to Python
* Add a new `TensorExpression` implementation for matrix-related expression evaluations.
* Add Lazy Assignment for optimize the calculation of multiple expressions.
* Add `Function` to reconstruct the computation function.
  * PadFunc and PadGradFunc.
  * ContextProjectionForwardFunc and ContextProjectionBackwardFunc.
  * CosSimBackward and CosSimBackwardFunc.
  * CrossMapNormalFunc and CrossMapNormalGradFunc.
  * MulFunc.
* Add `AutoCompare` and `FunctionCompare`, which make it easier to write unittest for comparing gpu and cpu version of a function.
* Add `libpaddle_test_main.a` and remove the main function inside the test file.
* Support dense numpy vector in PyDataProvider2.
* Clean code base, remove some copy & paste codes before.
  * Extract RowBuffer class for SparseRowMatrix.
  * Clean GradientMachine's interface.
  * Try use `override` keyword in layer.
  * Simplify Evaluator::create, use `ClassRegister` to create Evaluator.
* Add md5 check when downloading demo's dataset.
* Add `paddle::Error` which intentially replace `LOG(FATAL)` in Paddle.

## Bug Fixes

* Add layer check for recurrent_group.
* Clang-format off on some cuda .cc files.
* Fix LogActivation which is not defined.
* Fix bug when run test_layerHelpers multiple times.
* Fix protobuf size limit on seq2seq demo.
* Fix bug for dataprovider converter in GPU mode.
* Fix bug in GatedRecurrentLayer which only occurs in predicting or `job=test` mode.
* Fix bug for BatchNorm when testing more than models in test mode.
* Fix unit test of paramRelu.
* Fix some warning about CpuSparseMatrix.
* Fix MultiGradientMachine error if trainer_count > batch_size.
* Fix when async load data in PyDataProvider2.

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
