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
